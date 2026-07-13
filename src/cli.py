"""Console-script entry point for the training pipeline.

Mirrors running, from the project root:

    uv run python src/main.py --config config.yaml --data-dir /content/liver_data

All command-line arguments are forwarded to ``src/main.py``. Output (stdout
and stderr) is streamed live to the console AND mirrored to ``run.log`` in the
current working directory, so progress is visible in real time and a full
record is kept even if the cell hides output.
"""

import os
import subprocess
import sys


def train_cmd():
    cmd = ["uv", "run", "python", "src/main.py", *sys.argv[1:]]
    log_path = os.path.join(os.getcwd(), "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        rc = proc.wait()
    sys.exit(rc)
