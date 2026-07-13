"""CLI entry point: ``simple_ai_autoresearch_train``.

Autonomous config-search trainer for the simple_ai classification pipeline.
The agent only edits ``config.yaml``; training is measured by validation loss.

Examples
--------
    # Local Ollama (Colab T4), dataset already present
    simple_ai_autoresearch_train --local --data-dir /content/dataset --runs 12

    # OpenRouter free tier
    export OPENROUTER_API_KEY=sk-or-...
    simple_ai_autoresearch_train --data-dir /content/dataset \\
        --model meta-llama/llama-3.1-8b-instruct:free --runs 12

    # Download the dataset from Google Drive first
    simple_ai_autoresearch_train --local --gdown-id 1LNkF... --runs 12
"""

import argparse
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="simple_ai_autoresearch_train",
        description="Autonomously improve classification training by editing config.yaml.",
    )
    parser.add_argument("--data-dir", default=None, help="Dataset directory for training.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path (edited in place).")
    parser.add_argument("--experiments", default="experiments.tsv", help="TSV log of all runs.")
    parser.add_argument("--model", default=None, help="LLM model id (default depends on --local).")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default OpenRouter). Ignored with --local.",
    )
    parser.add_argument("--local", action="store_true", help="Use a local Ollama server.")
    parser.add_argument(
        "--unload-between-runs",
        action="store_true",
        help="Unload the Ollama model between runs to free VRAM during training.",
    )
    parser.add_argument("--timeout", type=int, default=700, help="Training timeout (seconds).")
    parser.add_argument("--runs", type=int, default=10, help="Number of optimization runs.")
    parser.add_argument("--gdown-id", default=None, help="Google Drive file id to download first.")
    parser.add_argument("--data-name", default=None, help="Dataset name used for archive naming.")
    parser.add_argument("--repo-root", default=None, help="Repo root (default: current dir).")

    args = parser.parse_args(argv)

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
