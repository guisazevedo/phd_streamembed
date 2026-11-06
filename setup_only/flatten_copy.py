#!/usr/bin/env python3
"""Copy every file from a folder tree into a single flat destination folder.

Edit the CONFIG section below, then run:
    python flatten_copy.py

Name collisions are resolved by appending _1, _2, ... to the file name.
Files are only copied, never moved or deleted.
"""
import shutil
from pathlib import Path

# ---------------------------------------------------------------- CONFIG ----
SOURCE = Path("/home/guisazevedo/Documents/00_arbeit/sa/data/bracs_wsi/BRACS_WSI")  # root folder to scan (recursively)
DEST = Path("../data")  # single folder to put all files in
EXTENSIONS = [".svs"]  # e.g. [".jpg", ".pdf"]; empty list = copy all files
# -----------------------------------------------------------------------------


def unique_path(dest_dir: Path, name: str) -> Path:
    target = dest_dir / name
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    i = 1
    while True:
        target = dest_dir / f"{stem}_{i}{suffix}"
        if not target.exists():
            return target
        i += 1


def flatten_copy(source: Path, dest: Path, extensions=None) -> int:
    source, dest = source.resolve(), dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions or []}
    count = 0

    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if dest in path.resolve().parents:  # skip files already in the destination
            continue
        if exts and path.suffix.lower() not in exts:
            continue
        shutil.copy2(path, unique_path(dest, path.name))
        count += 1
    return count


def main():
    if not SOURCE.is_dir():
        raise SystemExit(f"source is not a directory: {SOURCE}")

    n = flatten_copy(SOURCE, DEST, EXTENSIONS)
    print(f"Copied {n} files to {DEST}")


if __name__ == "__main__":
    main()
