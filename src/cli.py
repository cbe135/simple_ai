"""Console-script entry point for the training pipeline.

Mirrors running, from the project root:

    uv run python src/main.py --config config.yaml --data-dir /content/liver_data \
        > run.log 2>&1; cat run.log

All command-line arguments are forwarded to ``src/main.py``. Output (stdout
and stderr) is captured to ``run.log`` in the current working directory and
then printed, so progress/tracebacks are never lost even when the subprocess
is otherwise hidden (e.g. inside a Colab ``!`` cell).
"""

import os
import subprocess
import sys


def train_cmd():
    cmd = ["uv", "run", "python", "src/main.py", *sys.argv[1:]]
    log_path = os.path.join(os.getcwd(), "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    with open(log_path) as log:
        sys.stdout.write(log.read())
    sys.exit(proc.returncode)
