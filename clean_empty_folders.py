#!/usr/bin/env python3
"""
clean_empty_folders.py

Recursively deletes directories that are empty or contain only ignorable files
(.DS_Store, metadata.json). Logs to logs/clean_empty_folders.log.
"""

from pathlib import Path
from typing import Iterable, Optional
import logging
import os
from tqdm import tqdm

# Centralized logs directory at project root
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"{Path(__file__).stem}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # only log errors and above
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(LOG_FILE) for h in logger.handlers):
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setLevel(logging.ERROR)  # ensure handler filters to errors
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# Files that do not count as content when deciding if a folder is empty
IGNORABLE = {".DS_Store", "metadata.json", "Thumbs.db"}


def clean_empty_folders(target_directory: Path, *, dry_run: bool = False, extra_ignorable: Optional[Iterable[str]] = None) -> int:
    """
    Walk the tree bottom-up and remove directories that are empty or contain only ignorable files.
    Supports a dry-run mode and optional extra ignorable filenames.
    Returns the number of directories removed.
    """
    target_directory = Path(target_directory)

    ignorable = set(IGNORABLE)
    if extra_ignorable:
        ignorable.update(extra_ignorable)

    # Stream directories bottom-up using os.walk (avoids holding all paths in memory)
    removed_count = 0
    # We don't have a fixed total without a full pre-scan, so let tqdm be indeterminate
    for root, dirs, files in os.walk(target_directory, topdown=False):
        directory = Path(root)
        # progress display (one tick per directory)
        tqdm.write(f"")  # ensure tqdm initializes cleanly in some terminals
        with tqdm(total=None, desc="🧹 Cleaning empty folders", unit="dir", leave=False) as _pbar:
            break  # create once
    # Re-run the actual walk now that tqdm is initialized
    with tqdm(total=None, desc="🧹 Cleaning empty folders", unit="dir", leave=False) as pbar:
        for root, dirs, files in os.walk(target_directory, topdown=False):
            directory = Path(root)
            pbar.update(1)

            # Prevent deletion of the root directory itself
            if directory == target_directory:
                continue
            try:
                # Collect entries and filter out ignorable files
                entries = list(directory.iterdir())
                non_ignorable = [e for e in entries if not (e.is_file() and e.name in ignorable)]

                if not non_ignorable:
                    if not dry_run:
                        for e in entries:
                            if e.is_file() and e.name in ignorable:
                                try:
                                    e.unlink()
                                except Exception as error:
                                    logger.error(f"Could not remove ignorable file {e}: {error}")
                        try:
                            directory.rmdir()
                            removed_count += 1
                            logger.error(f"Removed empty folder: {directory}")
                        except Exception as error:
                            logger.error(f"Could not remove directory {directory}: {error}")
                    else:
                        # Dry-run: count as would-remove, but make no changes
                        removed_count += 1

            except Exception as error:
                # Skip any directory we cannot remove, but log it for diagnostics
                logger.error(f"Skipped directory {directory}: {error}")

    if removed_count == 0:
        print(f"\n✅ No empty folders found in {target_directory}")
    else:
        action = "Would remove" if dry_run else "Removed"
        print(f"\n🗑️ {action} {removed_count} empty folder(s)")
    return removed_count