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
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List

from tqdm import tqdm

import logging

# Centralized logs directory at project root
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"{Path(__file__).stem}.log"

# Configure per-module file logger (idempotent) - ERROR-only
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # only log errors and above
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(LOG_FILE) for h in logger.handlers):
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setLevel(logging.ERROR)  # ensure handler filters to errors
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


# Supported tar-like archive suffixes (extendable)
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".tzst")

# Skip common noisy directories/files during search
SKIP_DIRS = {"__MACOSX", ".Trash", ".Trashes", ".Spotlight-V100"}
SKIP_FILES = {".DS_Store", "Thumbs.db"}


# ---------------------- Core helpers ----------------------

def find_archives(root: Path) -> List[Path]:
    """Recursively find archives with known tar-like suffixes under root."""
    root = Path(root)
    archives: List[Path] = []
    lower_suffixes = tuple(s.lower() for s in ARCHIVE_SUFFIXES)
    for dirpath, dirnames, filenames in os.walk(root):
        # prune noisy/system dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            if name in SKIP_FILES:
                continue
            lname = name.lower()
            if lname.endswith(lower_suffixes):
                archives.append(base / name)
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
            logger.debug(f"Skipped member {member.name}: {error}")
    return files_extracted


def _is_zip(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def extract_archive(archive_path: Path, target_dir: Path, pbar=None) -> int:
    """Open an archive and extract safely. Uses tar for tar-like, shutil for zip."""
    if _is_zip(archive_path):
        # Use zipfile to count members, then extract; update pbar by count
        try:
            with zipfile.ZipFile(archive_path, mode="r") as zf:
                file_members = [zi for zi in zf.infolist() if not zi.is_dir()]
                count = len(file_members)
                zf.extractall(path=target_dir)
                if pbar is not None and count:
                    pbar.update(count)
                return count
        except Exception as err:
            print(f"❌ Failed extracting ZIP {archive_path}: {err}", file=sys.stderr)
            logger.error(f"Failed extracting ZIP {archive_path}: {err}")
            return 0
    else:
        with tarfile.open(archive_path, mode="r:*") as tf:
            return _safe_extract(tf, target_dir, pbar=pbar)


def _count_files_in_archives(archives: List[Path]) -> int:
    """Return the total number of regular files contained in all archives.

    Uses a streaming count for tar and zip; skips counting when there are many
    archives to avoid a costly pre-pass (progress bar will be indeterminate).
    """
    if len(archives) > 20:
        return 0  # skip pre-count; we'll show an indeterminate progress bar

    total = 0
    for arc in archives:
        try:
            if _is_zip(arc):
                with zipfile.ZipFile(arc, mode="r") as zf:
                    total += sum(1 for zi in zf.infolist() if not zi.is_dir())
            else:
                with tarfile.open(arc, mode="r:*") as tf:
                    total += sum(1 for m in tf if m.isfile())
        except Exception:
            continue
    return total


def extract_all(source_dir: Path, target_dir: Path, workers: int = 1) -> int:
    """Extract all .tar/.tar.gz/.tgz archives found under source_dir into target_dir.

    Returns the number of files successfully extracted.
    """
    source_dir = Path(source_dir).expanduser()
    target_dir = Path(target_dir).expanduser()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Prevent extracting into a subdirectory of source (can cause recursion and re-scans)
    try:
        if str(target_dir.resolve(strict=False)).startswith(str(source_dir.resolve(strict=False)) + os.sep):
            raise ValueError("Target directory must not be inside the source directory.")
    except Exception:
        pass

    archives = find_archives(source_dir)
    if not archives:
        print(f"⚠️  No archives found in {source_dir}")
        logger.debug(f"No archives found in {source_dir}")
        return 0

    total_members = _count_files_in_archives(archives)
    total_extracted = 0

    desc = "📦 Extracting files" if total_members else "📦 Extracting (counting…)"
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with tqdm(total=(total_members or None), desc=desc, unit="file", leave=False) as pbar:
        if workers and workers > 1:
            # Parallel extraction across archives; pbar is shared and thread-safe
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {pool.submit(extract_archive, arc, target_dir, pbar): arc for arc in archives}
                for fut in as_completed(future_map):
                    arc = future_map[fut]
                    try:
                        extracted = fut.result()
                        total_extracted += int(extracted or 0)
                    except tarfile.ReadError as err:
                        print(f"❌ Not a valid tar archive: {arc} ({err})", file=sys.stderr)
                        logger.error(f"Not a valid tar archive: {arc} ({err})")
                    except Exception as err:
                        print(f"❌ Failed extracting {arc}: {err}", file=sys.stderr)
                        logger.error(f"Failed extracting {arc}: {err}")
        else:
            # Serial fallback
            for arc in archives:
                try:
                    pbar.set_postfix_str(arc.name)
                    total_extracted += extract_archive(arc, target_dir, pbar=pbar)
                except tarfile.ReadError as err:
                    print(f"❌ Not a valid tar archive: {arc} ({err})", file=sys.stderr)
                    logger.error(f"Not a valid tar archive: {arc} ({err})")
                except Exception as err:
                    print(f"❌ Failed extracting {arc}: {err}", file=sys.stderr)
                    logger.error(f"Failed extracting {arc}: {err}")

    return total_extracted


# ---------------------- CLI entry point ----------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract archives (.tar/.tgz/.tar.gz/.bz2/.xz/.zst and .zip) into a target folder.")
    p.add_argument("--source", required=True, help="Folder to search for archives")
    p.add_argument("--target", required=True, help="Destination folder for extraction")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel workers to extract archives (default: 1).")
    return p.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        print("Usage: extract_archives.py --source <folder> --target <folder>")
        sys.exit(2)
    args = _parse_args()
    count = extract_all(Path(args.source), Path(args.target), workers=int(args.workers))
    print(f"✅ Extracted {count} file(s)")
    logger.info(f"Extracted {count} file(s) from archives in CLI run")


if __name__ == "__main__":
    main()
