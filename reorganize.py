#!/usr/bin/env python3
"""
reorganize.py

Moves media files based on their chosen_timestamp in the catalog.
Files are organized into OUTPUT_DIRECTORY/YYYY/ or OUTPUT_DIRECTORY/YYYY/MM/ (keeping the original filename) based on user choice.
"""

import sqlite3
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from catalog import (
    TABLE_NAME,
    COL_RENAMED,
    COL_CHOSEN,
    batch_update_renamed_filepath,
)

import logging

from concurrent.futures import ThreadPoolExecutor, as_completed

# I/O worker threads for moving files
MOVE_WORKERS = 8

# Centralized logs directory at project root
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"{Path(__file__).stem}.log"

# Configure per-module file logger (ERROR only)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # only log errors and above
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(LOG_FILE) for h in logger.handlers):
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setLevel(logging.ERROR)  # ensure handler filters to errors
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


def make_unique_path(dest_dir: Path, filename: str) -> Path:
    """
    If 'dest_dir/filename' exists, append ' (1)', ' (2)', ... before the extension.
    """
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dest_dir / f"{base}{ext}"
    counter = 1
    while candidate.exists():
        candidate = dest_dir / f"{base} ({counter}){ext}"
        counter += 1
    return candidate


def safe_move(pair):
    src, dst = pair
    try:
        # Use shutil.move to support cross-device moves
        shutil.move(str(src), str(dst))
        return (src, dst, True, None)
    except Exception as e:
        return (src, dst, False, e)


def reorganize_files(connection: sqlite3.Connection, output_directory: Path, mode: Optional[str] = None) -> None:
    """
    Read chosen_timestamp from the catalog and move/rename files into
    output_directory/YYYY/MM/ (or YYYY/) while keeping the original filename.

    Args:
        connection: Open SQLite connection.
        output_directory: Destination base directory.
        mode: Optional layout selector. Accepts:
              - "yyyy"    → organize into YYYY/
              - "yyyy-mm" → organize into YYYY/MM/
              If None or invalid, the function will prompt interactively.
    """
    # Determine layout from mode flag, if provided
    organize_by_month: Optional[bool] = None
    if mode in ("yyyy", "YYYY"):
        organize_by_month = False
    elif mode in ("yyyy-mm", "YYYY-MM"):
        organize_by_month = True

    # Ask how to organize: year only or year/month (only if mode not supplied)
    if organize_by_month is None:
        print("📁 Organize files into:")
        print("  [1] Year only (YYYY)")
        print("  [2] Year and month (YYYY/MM)")
        choice = input("Select 1 or 2: ").strip()
        organize_by_month = (choice == "2")

    cursor = connection.cursor()
    cursor.execute(
        f"SELECT {COL_RENAMED}, {COL_CHOSEN} FROM {TABLE_NAME} WHERE {COL_RENAMED} IS NOT NULL"
    )
    rows = cursor.fetchall()

    # 1) Plan moves
    pairs = []
    for renamed_filepath_str, chosen_timestamp_str in tqdm(rows, desc="🧭 Planning moves", unit="file", leave=False):
        if not renamed_filepath_str or not chosen_timestamp_str:
            continue
        src = Path(renamed_filepath_str)
        if not src.exists():
            continue

        # chosen_str example: "2023-07-14_21-05-09" or "2023-07-14_21-05-09-12"
        year  = chosen_timestamp_str[0:4]
        month = chosen_timestamp_str[5:7]

        destination_dir = (output_directory / year / month) if organize_by_month else (output_directory / year)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # keep original filename, but avoid clobbering
        dst = make_unique_path(destination_dir, src.name)

        if dst.resolve() == src.resolve():
            continue

        pairs.append((src, dst))

    if not pairs:
        print("Nothing to move.")
        return

    # 2) Execute moves concurrently (I/O-bound)
    updates = []
    errors = 0
    with ThreadPoolExecutor(max_workers=MOVE_WORKERS) as pool:
        futures = [pool.submit(safe_move, pair) for pair in pairs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="📂 Moving files", unit="file", leave=False):
            src, dst, ok, err = fut.result()
            if ok:
                updates.append((str(dst), str(src)))  # (new, old) for DB
            else:
                errors += 1
                logger.error(f"❌ Failed to move {src} → {dst}: {err}")

    # 3) Batch update DB in a single call
    if updates:
        try:
            batch_update_renamed_filepath(connection, updates)
        except Exception as e:
            logger.error(f"DB batch update failed for {len(updates)} items: {e}")

    if errors:
        print(f"Completed with {errors} error(s).")
