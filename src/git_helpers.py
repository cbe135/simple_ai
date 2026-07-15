"""Console-script helper: reconcile a divergent branch via rebase.

Equivalent to running, in the project root:

    git stash
    git pull --rebase
    git stash pop

but ``git stash`` / ``git stash pop`` only run when there are actually
uncommitted *tracked* changes. Untracked files (e.g. ``.opencode/``) survive a
rebase/merge untouched, so stashing them is unnecessary and would otherwise make
``git stash pop`` fail with "No stash entries found".
"""

import subprocess
import sys


def _run(cmd):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def main():
    # Detect uncommitted tracked changes; ignore untracked `??` lines.
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    has_tracked_changes = any(
        line and not line.startswith("??") for line in status.splitlines()
    )

    stashed = False
    if has_tracked_changes:
        stashed = _run(["git", "stash"]) == 0
    try:
        rc = _run(["git", "pull", "--rebase"])
        if rc != 0:
            print("git pull --rebase failed; aborting.", file=sys.stderr)
            return rc
    finally:
        if stashed:
            _run(["git", "stash", "pop"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
