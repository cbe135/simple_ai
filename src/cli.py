"""Console-script entry point for the training pipeline.

Mirrors running, from the project root:

    uv run python src/main.py --config config.yaml --data-dir /content/liver_data

All command-line arguments are forwarded to ``src/main.py``. Output (stdout
and stderr) is streamed live to the console AND mirrored to ``run.log`` in the
current working directory, so progress is visible in real time and a full
record is kept even if the cell hides output.
"""

import os
import pty
import select
import subprocess
import sys


def train_cmd():
    # Run the child under a pseudo-terminal so its stdout/stderr are a real TTY.
    # tqdm then overwrites the progress bar in place (instead of emitting one
    # line per update, its fallback for non-interactive streams). Output is
    # still teed to run.log for a full record.
    master, slave = pty.openpty()
    log_path = os.path.join(os.getcwd(), "run.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            ["uv", "run", "python", "src/main.py", *sys.argv[1:]],
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        os.close(slave)
        try:
            while True:
                r, _, _ = select.select([master], [], [])
                data = os.read(master, 4096)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                log.write(data.decode(errors="replace"))
                log.flush()
        finally:
            os.close(master)
    sys.exit(proc.wait())
