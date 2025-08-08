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
from tqdm import tqdm
from catalog import update_renamed_filepath

import logging

# Ensure a logs directory exists next to this script
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = LOGS_DIR / "renaming.log"
handler = logging.FileHandler(log_file, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


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
    cursor.execute("SELECT renamed_filepath, chosen_date FROM media WHERE renamed_filepath IS NOT NULL")
    rows = cursor.fetchall()

    for renamed_filepath_str, chosen_date_str in rows:
        media_path = Path(renamed_filepath_str)

        if not media_path.exists():
            continue
        if not chosen_date_str:
            continue

        # chosen_str example: "2023-07-14_21-05-09" or "2023-07-14_21-05-09-12"
        year  = chosen_date_str[0:4]
        month = chosen_date_str[5:7]

        if organize_by_month:
            destination_dir = output_directory / year / month
        else:
            destination_dir = output_directory / year
        destination_dir.mkdir(parents=True, exist_ok=True)

        new_path = destination_dir / media_path.name

        if new_path != media_path:
            try:
                shutil.move(str(media_path), str(new_path))
                # Update database with the new renamed path (point renamed_filepath to new path)
                update_renamed_filepath(connection, media_path, new_path)
            except Exception as error:
                logger.error(f"❌ Failed to move {media_path.name}: {error}")
