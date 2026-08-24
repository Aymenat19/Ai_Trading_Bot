"""
sync_archive.py — push trade_archive.json to GitHub so viewer_app.py, deployed on
Streamlit Community Cloud, can display it. The scanning bot only runs locally
(app.py); Community Cloud never scans and never writes back (see viewer_app.py's
docstring), so this one-way sync is the only thing keeping the cloud dashboard
current.

Commits + pushes only when the file actually changed, so idle periods don't create
empty commits. Safe to run frequently (e.g. every 15 min via cron/launchd); it's a
no-op if there's nothing new since the last sync.

Usage:
    python scripts/sync_archive.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FILE = "trade_archive.json"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def main() -> int:
    if not (REPO_ROOT / FILE).exists():
        print(f"{FILE} not found — nothing to sync.")
        return 0

    # --porcelain catches both modifications to a tracked file and a file that
    # has never been committed yet ("??") — `git diff` alone misses the latter.
    status = run(["git", "status", "--porcelain", "--", FILE])
    if not status.stdout.strip():
        print(f"{FILE} unchanged — skipping sync.")
        return 0

    add = run(["git", "add", FILE])
    if add.returncode != 0:
        print(f"git add failed: {add.stderr}", file=sys.stderr)
        return 1

    commit = run(["git", "commit", "-m", "Sync trade_archive.json"])
    if commit.returncode != 0:
        print(f"git commit failed: {commit.stderr}", file=sys.stderr)
        return 1

    push = run(["git", "push"])
    if push.returncode != 0:
        print(f"git push failed: {push.stderr}", file=sys.stderr)
        return 1

    print(f"Synced {FILE} to GitHub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
