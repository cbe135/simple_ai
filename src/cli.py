"""Console-script entry point for the training pipeline.

Mirrors running, from the project root:

    uv run python src/main.py --config config.yaml --data-dir /content/liver_data

All command-line arguments are forwarded to ``src/main.py``. Log output on
stdout is streamed live to the console AND mirrored to ``run.log`` in the
current working directory, so progress is visible in real time and a full
record is kept even if the cell hides output.

Note: stderr is intentionally INHERITED (not piped/redirected). The training
code points its tqdm progress bars at stderr so they render in place on the
real Colab/Jupyter cell TTY. If stderr were merged into the stdout pipe,
tqdm would detect a non-interactive stream and either print one line per
update or, under a pseudo-terminal, emit carriage-return-only updates that
Colab does not render incrementally -- making the bar effectively invisible.
"""

import os
import subprocess
import sys


def train_cmd():
    log_path = os.path.join(os.getcwd(), "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            ["uv", "run", "python", "src/main.py", *sys.argv[1:]],
            # stderr is inherited so tqdm renders in place on the cell TTY.
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        rc = proc.wait()
    sys.exit(rc)
