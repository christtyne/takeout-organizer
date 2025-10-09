#!/usr/bin/env python3
"""
Google Takeout Photo Organizer – Archive Extractor (pure Python)

- Recursively scans a source folder for .tar / .tar.gz / .tgz files
- Extracts each archive into the chosen target directory
- Uses a safe extraction routine to avoid path traversal
- Provides both a callable API (extract_all) and a CLI entry point
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm


# ---------------------- Core helpers ----------------------

def find_archives(root: Path) -> List[Path]:
    """Recursively find .tar / .tar.gz / .tgz files under root."""
    patterns = (".tar", ".tar.gz", ".tgz")
    archives: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower().endswith(patterns):
            archives.append(path)
    return archives


def _is_within_directory(directory: Path, target: Path) -> bool:
    """Ensure target is inside directory (prevents path traversal)."""
    try:
        directory = directory.resolve(strict=False)
        target = target.resolve(strict=False)
    except Exception:
        directory = Path(os.path.abspath(str(directory)))
        target = Path(os.path.abspath(str(target)))
    return str(target).startswith(str(directory))


def _safe_extract(tar: tarfile.TarFile, dest: Path, pbar=None) -> int:
    """Safely extract tar members into dest, rejecting traversal/absolute paths.

    Returns the number of regular files extracted. Updates tqdm if provided.
    """
    files_extracted = 0
    for member in tar.getmembers():
        name = member.name
        # Sanitize absolute paths and parent traversal
        if name.startswith("/") or ".." in Path(name).parts:
            member.name = Path(name).name  # keep basename only
        target_path = dest / member.name
        if not _is_within_directory(dest, target_path):
            continue
        try:
            tar.extract(member, path=dest)
            if member.isfile():
                files_extracted += 1
                if pbar is not None:
                    pbar.update(1)
        except Exception as error:
            print(f"⚠️  Skipped member {member.name}: {error}", file=sys.stderr)
    return files_extracted


def extract_archive(archive_path: Path, target_dir: Path, pbar=None) -> int:
    """Open a tar archive (auto-detect compression) and extract safely.

    Returns the number of regular files extracted from this archive.
    """
    with tarfile.open(archive_path, mode="r:*") as tf:
        return _safe_extract(tf, target_dir, pbar=pbar)


def _count_files_in_archives(archives: List[Path]) -> int:
    """Return the total number of regular files contained in all tar archives."""
    total = 0
    for arc in archives:
        try:
            with tarfile.open(arc, mode="r:*") as tf:
                total += sum(1 for m in tf.getmembers() if m.isfile())
        except Exception:
            # If we can't read this archive now, skip counting; extraction will still try
            continue
    return total


def extract_all(source_dir: Path, target_dir: Path) -> int:
    """Extract all .tar/.tar.gz/.tgz archives found under source_dir into target_dir.

    Returns the number of files successfully extracted.
    """
    source_dir = Path(source_dir).expanduser()
    target_dir = Path(target_dir).expanduser()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    archives = find_archives(source_dir)
    if not archives:
        print(f"⚠️  No archives found in {source_dir}")
        return 0

    total_members = _count_files_in_archives(archives)
    total_extracted = 0

    desc = "📦 Extracting files" if total_members else "📦 Extracting (counting…)"
    with tqdm(total=total_members or None, desc=desc, unit="file") as pbar:
        for arc in archives:
            try:
                total_extracted += extract_archive(arc, target_dir, pbar=pbar)
            except tarfile.ReadError as err:
                print(f"❌ Not a valid tar archive: {arc} ({err})", file=sys.stderr)
            except Exception as err:
                print(f"❌ Failed extracting {arc}: {err}", file=sys.stderr)

    return total_extracted


# ---------------------- CLI entry point ----------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract .tar/.tgz/.tar.gz archives into a target folder.")
    p.add_argument("--source", required=True, help="Folder to search for archives")
    p.add_argument("--target", required=True, help="Destination folder for extraction")
    return p.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        print("Usage: extract_archives.py --source <folder> --target <folder>")
        sys.exit(2)
    args = _parse_args()
    count = extract_all(Path(args.source), Path(args.target))
    print(f"✅ Extracted {count} file(s)")


if __name__ == "__main__":
    main()
