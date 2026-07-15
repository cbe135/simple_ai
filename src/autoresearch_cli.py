"""CLI entry point: ``simple_ai_autoresearch_train``.

Autonomous config-search trainer for the simple_ai classification pipeline.
The agent only edits ``config.yaml``; training is measured by validation loss.

Examples
--------
    # Local Ollama is the default (Colab T4). Run setup first:
    simple_ai_autoresearch_setup
    # ...then optimize (dataset already present):
    simple_ai_autoresearch_train --data-dir /content/dataset --runs 12

    # OpenRouter free tier (opt in with --remote)
    export OPENROUTER_API_KEY=sk-or-...
    simple_ai_autoresearch_train --remote --data-dir /content/dataset \\
        --model meta-llama/llama-3.1-8b-instruct:free --runs 12

    # Download the dataset from Google Drive first
    simple_ai_autoresearch_train --gdown-id 1LNkF... --runs 12
"""

import argparse
import logging
import os
import sys
from logging import FileHandler, Formatter
from pathlib import Path


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    _log_path = Path.cwd() / "autoresearch.log"
    _file_handler = FileHandler(_log_path)
    _file_handler.setFormatter(
        Formatter(
            "[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(_file_handler)

    parser = argparse.ArgumentParser(
        prog="simple_ai_autoresearch_train",
        description="Autonomously improve classification training by editing config.yaml.",
    )
    parser.add_argument("--data-dir", default=None, help="Dataset directory for training.")
    parser.add_argument(
        "--modality",
        required=True,
        choices=["ct", "mri", "xray", "color"],
        help="Imaging modality (ct | mri | xray | color); passed through to training.",
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path (edited in place).")
    parser.add_argument("--experiments", default="experiments.tsv", help="TSV log of all runs.")
    parser.add_argument("--model", default=None, help="LLM model id (default depends on --local).")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default OpenRouter). Ignored with --local.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Use a remote Ollama server (with --local). Full OpenAI-compatible URL "
        "including /v1, e.g. https://abc.loca.lt/v1. Skips starting a local server.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use a local Ollama server (this is the default).",
    )
    parser.add_argument(
        "--remote",
        action="store_false",
        dest="local",
        help="Use OpenRouter instead of the local Ollama server (needs OPENROUTER_API_KEY).",
    )
    parser.set_defaults(local=True)
    parser.add_argument(
        "--unload-between-runs",
        action="store_true",
        help="Unload the Ollama model between runs to free VRAM during training.",
    )
    parser.add_argument("--timeout", type=int, default=700, help="Training timeout (seconds).")
    parser.add_argument("--runs", type=int, default=10, help="Number of optimization runs.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for this wrapper and the training subprocess "
        "(via SIMPLE_AI_LOG_LEVEL). Default: INFO.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the persistent transformed-data cache before each training run.",
    )
    parser.add_argument("--gdown-id", default=None, help="Google Drive file id to download first.")
    parser.add_argument("--data-name", default=None, help="Dataset name used for archive naming.")
    parser.add_argument("--repo-root", default=None, help="Repo root (default: current dir).")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Ollama models directory (default on Colab: /content/drive/MyDrive/"
        "ollama_models; otherwise ~/.ollama/models). Can also be set via $OLLAMA_MODELS. "
        "Use a Google Drive mount to reuse weights across sessions without re-downloading.",
    )

    args = parser.parse_args(argv)

    # Apply the chosen log level to this wrapper and forward it to the training
    # subprocess (main.py reads SIMPLE_AI_LOG_LEVEL for its own basicConfig).
    log_level = (args.log_level or "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    os.environ["SIMPLE_AI_LOG_LEVEL"] = log_level

    if args.gdown_id and not args.data_dir:
        args.data_dir = str(Path(args.data_dir or "/content/dataset"))

    if args.gdown_id:
        from .prepare_data import main as prepare_main

        prep_argv = ["--data-dir", str(args.data_dir), "--gdown-id", args.gdown_id]
        if args.data_name:
            prep_argv += ["--data-name", args.data_name]
        prepare_main(prep_argv)

    from .autoresearch import run as ar_run

    ar_run(vars(args))


if __name__ == "__main__":
    main()
