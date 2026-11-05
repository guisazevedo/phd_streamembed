#!/usr/bin/env python3
"""Upload slide files to the remote workstation with rsync (outside PyCharm's sync).

Edit the CONFIG section below, then run:
    python rsync_upload.py            # real upload
    python rsync_upload.py --dry-run  # only list what would be transferred

Only files matching EXTENSIONS are sent; the folder structure under SOURCE is kept.
Re-running is cheap: rsync skips files that are already complete on the remote,
and --partial lets an interrupted upload resume.
"""
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- CONFIG ----
SOURCE = Path("./data")
REMOTE_HOST = "guisazevedo@workstation"  # ssh target, or a Host alias from ~/.ssh/config
REMOTE_DEST = "/home/guisazevedo/Projects/01_accamargo/phd_streamembed/data"  # outside the PyCharm project folder
EXTENSIONS = [".svs"]  # empty list = send all files
SSH_PORT = None  # e.g. 2222; None = default
# -----------------------------------------------------------------------------


def build_command(dry_run: bool) -> list[str]:
    cmd = ["rsync", "-avh", "--partial", "--progress"]
    if dry_run:
        cmd.append("--dry-run")
    if SSH_PORT:
        cmd += ["-e", f"ssh -p {SSH_PORT}"]
    if EXTENSIONS:
        cmd.append("--include=*/")  # keep directory traversal
        cmd += [f"--include=*{e}" for e in EXTENSIONS]
        cmd += ["--exclude=*", "--prune-empty-dirs"]
    # trailing slash on the source: copy its contents, not the folder itself
    cmd += [f"{SOURCE}/", f"{REMOTE_HOST}:{REMOTE_DEST}/"]
    return cmd


def main():
    if shutil.which("rsync") is None:
        raise SystemExit("rsync is not installed")
    if not SOURCE.is_dir():
        raise SystemExit(f"source is not a directory: {SOURCE}")

    cmd = build_command(dry_run="--dry-run" in sys.argv)
    print(" ".join(cmd))
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
