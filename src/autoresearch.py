"""Autonomous config-search loop for the simple_ai classification pipeline.

The agent improves training by editing ONLY ``config.yaml``. A single run is:

  1. Ask the LLM for a new ``config.yaml`` (given the current one + history).
  2. Run ``uv run python src/main.py --config config.yaml --data-dir <dir>`` as a
     subprocess, capturing stdout.
  3. Parse the final ``Validation loss: <float>`` line (lower is better).
  4. If the run improved on the best known validation loss, keep it (commit
     ``config.yaml``); otherwise discard it (``git checkout config.yaml``).
     Every run is appended to ``experiments.tsv``.

The LLM client is OpenAI-compatible: OpenRouter by default, or a local Ollama
server when ``--local`` is passed. When running locally the Ollama server is
started on launch and shut down on completion or interrupt.

The training subprocess and the LLM call are fully sequential: phases never
overlap.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import re
import signal
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .autoresearch_paths import apply_models_dir

logger = logging.getLogger(__name__)

PROGRAM_MD = Path(__file__).with_name("program.md")
OLLAMA_PORT = 11434
VALID_OPTIMIZERS = {"adam", "adamw", "sgd"}
VALID_LOSSES = {"bce_with_logits", "cross_entropy"}


# --------------------------------------------------------------------------- #
# Config parsing / validation
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(path: Path, cfg: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def extract_yaml_block(text: str):
    """Return ``(config_dict, error)`` parsed from an LLM response.

    Accepts a fenced ```yaml block, or falls back to the whole response body.
    On any failure the first element is ``None`` and the second describes it.
    """
    if not text:
        return None, "empty LLM response"
    m = re.search(r"```(?:yaml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = m.group(1) if m else text
    try:
        cfg = yaml.safe_load(candidate)
    except yaml.YAMLError as e:
        return None, f"yaml parse error: {e}"
    if not isinstance(cfg, dict):
        return None, "config is not a YAML mapping"
    if "training" not in cfg:
        return None, "missing required 'training' section"
    # Validate the optimizer / loss names so we fail fast on bad proposals.
    opt = ((cfg.get("training", {}) or {}).get("optimizer", {}) or {}).get("name")
    if opt is not None and str(opt).lower() not in VALID_OPTIMIZERS:
        return None, f"unsupported optimizer: {opt!r}"
    loss = ((cfg.get("training", {}) or {}).get("loss", {}) or {}).get("name")
    if loss is not None and str(loss).lower() not in VALID_LOSSES:
        return None, f"unsupported loss: {loss!r}"
    return cfg, None


def parse_validation_loss(stdout: str):
    """Return ``(val_loss, error)`` from captured training stdout.

    The pipeline prints a line like ``Validation loss: 0.123`` each epoch; we
    take the last occurrence (final epoch).
    """
    if not stdout:
        return None, "no stdout captured from training"
    matches = re.findall(r"Validation loss:\s*([0-9]+(?:\.[0-9]+)?)", stdout)
    if not matches:
        return None, "no 'Validation loss:' line found in training output"
    try:
        return float(matches[-1]), None
    except ValueError:
        return None, "could not parse validation loss float"


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible)
# --------------------------------------------------------------------------- #
def make_client(base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client, model: str, system: str, user: str, max_tokens: int = 4096) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Local Ollama lifecycle
# --------------------------------------------------------------------------- #
_ollama_proc = None


def _ollama_reachable() -> bool:
    """Return True if an Ollama server already answers on ``OLLAMA_PORT``."""
    try:
        with socket.create_connection(("localhost", OLLAMA_PORT), timeout=2):
            return True
    except OSError:
        return False


def start_ollama() -> None:
    global _ollama_proc
    if _ollama_reachable():
        logger.info("Ollama server already running; reusing it.")
        return
    if shutil.which("ollama") is None:
        raise SystemExit(
            "Ollama binary not found. Run `simple_ai_autoresearch_setup` first to "
            "install it, or install Ollama manually from https://ollama.com/download "
            "(and select a GPU runtime on Colab)."
        )
    _ollama_proc = subprocess.Popen(
        ["ollama", "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(60):
        try:
            subprocess.run(
                ["curl", "-s", f"http://localhost:{OLLAMA_PORT}"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("Ollama server did not become reachable in time")


def _stream_command(cmd, timeout=None) -> int:
    """Run a command and stream its combined (stdout+stderr) output to the log.

    Unlike ``subprocess.run(..., capture_output=True)`` this keeps the user
    informed with live progress (e.g. the multi-GB Ollama download) instead of
    appearing to hang with no output. Raises ``subprocess.CalledProcessError``
    on a non-zero exit code.
    """
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line:
            logger.info("[ollama] %s", line)
    proc.wait(timeout=timeout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def pull_model(model: str) -> None:
    """Pull ``model`` if not already present, streaming download progress."""
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model in (res.stdout or ""):
        logger.info("Model %s already present; skipping pull.", model)
        return
    logger.info("Pulling Ollama model %s (this may take a while, ~5GB)...", model)
    try:
        _stream_command(["ollama", "pull", model])
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"Failed to pull model {model}: {e}")


def verify_ollama_gpu(model: str, keep_alive: str = "5m") -> None:
    """Confirm Ollama runs ``model`` on the GPU, not CPU.

    Uses the non-interactive ``/api/generate`` REST endpoint (with
    ``keep_alive`` so the model stays resident long enough for ``ollama ps`` to
    report it) instead of the interactive ``ollama run`` CLI, which can appear
    to hang when there is no TTY.
    """
    logger.info("Warming up model %s to verify GPU usage...", model)
    payload = json.dumps(
        {
            "model": model,
            "prompt": "say hi",
            "stream": False,
            "keep_alive": keep_alive,
        }
    ).encode()
    url = f"http://localhost:{OLLAMA_PORT}/api/generate"
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode()
        try:
            reply = json.loads(body).get("response", "")
            logger.info("Model responded: %s", (reply or "").strip()[:80])
        except json.JSONDecodeError:
            pass
    except Exception as e:  # noqa: BLE001 - surface as a warning, keep going
        logger.warning("Model warm-up via API failed: %s", e)

    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
    out = ps.stdout or ""
    if "GPU" in out and "CPU" not in out:
        logger.info("Ollama is using the GPU. ✓")
    elif "CPU" in out:
        logger.warning(
            "Ollama is running the model on CPU only. On Colab T4 you must use a "
            "CUDA >= 12 runtime image (Runtime > Change runtime type > CUDA 12.x), "
            "otherwise Ollama silently falls back to CPU. Consider using OpenRouter "
            "(remove --local) if the GPU driver cannot be upgraded."
        )
    else:
        logger.info("GPU status could not be confirmed from `ollama ps`; continuing.")


def ensure_ollama_model(model: str) -> None:
    """Pull the model if missing, then warm it up and check GPU availability."""
    pull_model(model)
    verify_ollama_gpu(model)


def stop_ollama() -> None:
    global _ollama_proc
    if _ollama_proc is None:
        return
    try:
        subprocess.run(
            ["ollama", "stop", "--all"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    _ollama_proc.terminate()
    try:
        _ollama_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _ollama_proc.kill()
    _ollama_proc = None


def ollama_unload(model: str) -> None:
    subprocess.run(
        ["ollama", "stop", model],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ollama_warmup(client, model: str) -> None:
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Training subprocess + git helpers
# --------------------------------------------------------------------------- #
def run_training(repo_root: Path, config_name: str, data_dir, modality, timeout: int, clear_cache=False):
    cmd = ["uv", "run", "python", "src/main.py", "--config", config_name]
    if data_dir:
        cmd += ["--data-dir", str(data_dir)]
    if modality:
        cmd += ["--modality", str(modality)]
    if clear_cache:
        cmd += ["--clear-cache"]
    logger.info("Running training: %s", " ".join(cmd))
    logger.info("Streaming training output (this can take several minutes)...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    buf = []

    def _reader():
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                logger.info("[train] %s", line)
            buf.append(line)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        proc.kill()
        t.join()
        return None, None, f"training timed out after {timeout}s"
    rc = proc.wait()
    stdout = "\n".join(buf)
    if rc != 0:
        tail = "\n".join(buf[-20:])
        return None, None, f"training exited {rc}:\n{tail}"
    val_loss, err = parse_validation_loss(stdout)
    run_dir = None
    m = re.search(r"Run output directory:\s*(\S+)", stdout)
    if m:
        run_dir = m.group(1)
    return val_loss, run_dir, err


def _git(args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args), cwd=str(cwd), capture_output=True, text=True
    )


def git_commit(path: Path, msg: str) -> bool:
    cwd = path.parent
    _git(["add", str(path.name)], cwd)
    res = _git(["commit", "-m", msg], cwd)
    if res.returncode == 0:
        return True
    # Best-effort identity setup for fresh clones (e.g. Colab).
    if "Author identity" in (res.stderr or ""):
        _git(["config", "user.email", "autoresearch@local"], cwd)
        _git(["config", "user.name", "autoresearch"], cwd)
        res = _git(["commit", "-m", msg], cwd)
    if res.returncode != 0:
        logger.warning("git commit failed: %s", (res.stderr or "").strip())
        return False
    return True


def git_checkout(path: Path) -> None:
    _git(["checkout", "--", str(path.name)], path.parent)


# --------------------------------------------------------------------------- #
# Experiment bookkeeping
# --------------------------------------------------------------------------- #
def append_tsv(path: Path, run_idx: int, run_dir, val_loss, status: str, modified: str, notes: str) -> None:
    vl = "" if val_loss is None else f"{val_loss:.6f}"
    run_dir_s = run_dir or ""
    modified_s = (modified or "").replace("\t", " ").replace("\n", " ")
    notes = (notes or "").replace("\t", " ").replace("\n", " ")
    with open(path, "a") as f:
        f.write(f"{run_idx}\t{run_dir_s}\t{vl}\t{status}\t{modified_s}\t{notes}\n")


def diff_configs(old, new, prefix="") -> list:
    """Return a compact ``key: old -> new`` summary of leaf changes between configs."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return [f"{prefix}: {old!r} -> {new!r}"]
    changes = []
    for k in sorted(set(old) | set(new)):
        p = f"{prefix}.{k}" if prefix else str(k)
        ov = old.get(k)
        nv = new.get(k)
        if isinstance(ov, dict) and isinstance(nv, dict):
            changes.extend(diff_configs(ov, nv, p))
        elif ov != nv:
            changes.append(f"{p}: {ov!r} -> {nv!r}")
    return changes


def build_user_prompt(cfg_text: str, history) -> str:
    parts = [
        "Current config.yaml:\n```yaml\n" + cfg_text + "```\n",
    ]
    if history:
        parts.append("Recent experiments (oldest first):")
        for h in history[-8:]:
            vl = f"{h['val_loss']:.6f}" if h.get("val_loss") is not None else "n/a"
            note = h.get("notes") or ""
            parts.append(f"- status={h['status']} val_loss={vl} {note}")
    parts.append(
        "\nPropose an improved config.yaml. Respond with ONLY a single ```yaml "
        "fenced block containing the complete, valid config.yaml (do not trim or "
        "summarize sections). Keep num_classes consistent with the loss: use "
        "num_classes: 1 with loss bce_with_logits, or num_classes > 1 with loss "
        "cross_entropy. Lower validation loss is better."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def run(args: dict) -> None:
    load_dotenv()  # picks up .env (e.g. OPENROUTER_API_KEY)

    repo_root = Path(args.get("repo_root") or os.getcwd()).resolve()
    config_path = (repo_root / (args.get("config") or "config.yaml")).resolve()
    data_dir = args.get("data_dir")
    modality = args.get("modality")
    experiments_path = (repo_root / (args.get("experiments") or "experiments.tsv")).resolve()
    timeout = int(args.get("timeout", 700))
    num_runs = int(args.get("runs", 10))
    local = bool(args.get("local"))
    unload = bool(args.get("unload_between_runs"))

    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    system = PROGRAM_MD.read_text(encoding="utf-8")
    if not experiments_path.exists():
        with open(experiments_path, "w") as f:
            f.write("run\trun_dir\tval_loss\tstatus\tmodified\tnotes\n")

    client = None
    model = args.get("model")
    if local:
        if not model:
            model = "qwen2.5-coder:7b"
        # Point Ollama at the chosen store (Drive on Colab by default) before
        # starting the server or pulling, so weights are reused across sessions.
        models_dir = apply_models_dir(args.get("models_dir"))
        logger.info("Ollama models directory: %s", models_dir)
        ollama_base_url = args.get("ollama_base_url")
        if ollama_base_url:
            # Point at a remote Ollama server (e.g. exposed from another
            # instance via `simple_ai_autoresearch_serve --expose`). We don't
            # start or manage a local server in this case.
            base_url = ollama_base_url
            api_key = "ollama"
            logger.info("Using remote Ollama server at %s", base_url)
        else:
            start_ollama()
            ensure_ollama_model(model)
            base_url = f"http://localhost:{OLLAMA_PORT}/v1"
            api_key = "ollama"
            atexit.register(stop_ollama)

            def _on_signal(signum, frame):
                stop_ollama()
                sys.exit(1)

            signal.signal(signal.SIGINT, _on_signal)
            signal.signal(signal.SIGTERM, _on_signal)
    else:
        if not model:
            model = "meta-llama/llama-3.1-8b-instruct:free"
        base_url = args.get("base_url") or os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        )
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY is not set. Export it (or put it in .env), "
                "or pass --local to use a local Ollama server instead."
            )

    client = make_client(base_url, api_key)

    best = None  # (val_loss, config)
    history = []

    SEP = "#" * 80
    try:
        for i in range(num_runs):
            logger.info(SEP)
            logger.info("Run %d/%d", i + 1, num_runs)
            current_cfg = load_config(config_path)
            user_prompt = build_user_prompt(
                yaml.safe_dump(current_cfg, sort_keys=False), history
            )

            if local and unload:
                ollama_warmup(client, model)

            logger.info("Run %d/%d: asking LLM for a new config...", i + 1, num_runs)
            text = call_llm(client, model, system, user_prompt)
            cfg, err = extract_yaml_block(text)
            if err:
                logger.warning("Run %d skipped (LLM output invalid: %s)", i + 1, err)
                history.append({"status": "PARSE_ERROR", "notes": err, "val_loss": None, "run_dir": None, "modified": ""})
                append_tsv(experiments_path, i, None, None, "PARSE_ERROR", "", err)
                continue

            modified = "; ".join(diff_configs(current_cfg, cfg))
            save_config(config_path, cfg)

            if local and unload:
                ollama_unload(model)

            val_loss, run_dir, terr = run_training(
                repo_root, config_path.name, data_dir, modality, timeout,
                args.get("clear_cache", False),
            )
            if terr or val_loss is None:
                status = "ERROR"
                notes = terr or "no validation loss reported"
                logger.warning("Run %d failed: %s", i + 1, notes)
                git_checkout(config_path)
                history.append({"status": status, "notes": notes, "val_loss": None, "run_dir": run_dir, "modified": modified})
                append_tsv(experiments_path, i, run_dir, None, status, modified, notes)
                continue

            improved = best is None or val_loss < best[0]
            if improved:
                status = "baseline" if best is None else "improved"
                git_commit(config_path, f"autoresearch run {i}: val_loss={val_loss:.4f}")
                best = (val_loss, cfg)
                logger.info("Run %d %s: val_loss=%.4f", i + 1, status, val_loss)
            else:
                status = "regressed"
                logger.info(
                    "Run %d regressed: val_loss=%.4f (best=%.4f) -> discarding",
                    i + 1,
                    val_loss,
                    best[0],
                )
                git_checkout(config_path)

            history.append({"status": status, "notes": "", "val_loss": val_loss, "run_dir": run_dir, "modified": modified})
            append_tsv(experiments_path, i, run_dir, val_loss, status, modified, "")
    finally:
        if local:
            stop_ollama()

    if best is not None:
        logger.info("Best validation loss: %.4f", best[0])
    else:
        logger.warning("No successful run produced a validation loss.")
