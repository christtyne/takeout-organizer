#!/usr/bin/env python3
"""
reorganize.py

Moves media files based on their chosen_date in the catalog.
Files are organized into OUTPUT_DIRECTORY/YYYY/ or OUTPUT_DIRECTORY/YYYY/MM/ based on user choice
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm

def make_unique_path(
    destination_directory: Path,
    base_name: str,
    extension: str,
    original_file_path: Path = None
) -> Path:
    """
    Return a unique path in destination_directory:
    base_name+extension or base_name(n)+extension on collisions.
    """
    candidate = destination_directory / f"{base_name}{extension}"
    if not candidate.exists() or (
        original_file_path and candidate.samefile(original_file_path)
    ):
        return candidate

    index = 1
    while True:
        candidate = destination_directory / f"{base_name}({index}){extension}"
        if not candidate.exists():
            return candidate
        index += 1

def reorganize_files(connection: sqlite3.Connection, output_directory: Path) -> None:
    """
    Read chosen_date from the catalog and move/rename files into
    output_directory/YYYY/MM/ with a timestamp-based filename.
    """
    # Ask how to organize: year only or year/month
    print("📁 Organize files into:")
    print("  [1] Year only (YYYY)")
    print("  [2] Year and month (YYYY/MM)")
    choice = input("Select 1 or 2: ").strip()
    organize_by_month = (choice == "2")

    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT filepath, chosen_date
        FROM media
        WHERE chosen_date IS NOT NULL
        """
    )
    rows = cursor.fetchall()

    for filepath_str, chosen_date_str in tqdm(rows, desc="Reorganizing files", unit="file"):
        media_path = Path(filepath_str)
        try:
            timestamp = datetime.fromisoformat(chosen_date_str)
        except ValueError:
            # Skip entries with invalid date format
            continue

        year = timestamp.strftime("%Y")
        month = timestamp.strftime("%m")
        if organize_by_month:
            destination_dir = output_directory / year / month
        else:
            destination_dir = output_directory / year
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Build base filename with optional milliseconds
        base = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        if timestamp.microsecond:
            milliseconds = int(timestamp.microsecond / 1000)
            base = f"{base}-{milliseconds:02d}"

        extension = media_path.suffix.lower()
        new_path = make_unique_path(
            destination_dir, base, extension, original_file_path=media_path
        )

        try:
            shutil.move(str(media_path), str(new_path))
        except Exception as error:
            print(f"❌ Failed to move {media_path.name}: {error}")