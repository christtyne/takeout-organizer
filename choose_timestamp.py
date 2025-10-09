#!/usr/bin/env python3
"""
choose_timestamp.py

Choose the final 'chosen_date' for each media entry in the SQLite catalog
based on priority:
  1. EXIF CreateDate (most reliable)
  2. JSON photoTakenTime (fallback)
  3. Filename-parsed date
"""

import sqlite3
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from catalog import update_chosen_date

def choose_timestamp_for_all(connection: sqlite3.Connection) -> None:
    """
    For each media record, select the oldest available timestamp among
    exif_create_date, json_taken_date, and filename_parsed_date, then
    update the chosen_date field.
    """
    cursor = connection.cursor()
    cursor.execute("""
        SELECT filepath, exif_create_date, json_taken_date, filename_parsed_date
        FROM media
    """)
    rows = cursor.fetchall()

    for filepath, exif_str, json_str, filename_str in tqdm(rows, desc="Choosing timestamps", unit="file"):
        # Collect valid datetime objects with their source
        parsed_options = []  # list of tuples: (datetime, source)
        if exif_str:
            try:
                parsed_options.append((datetime.fromisoformat(exif_str), "exif"))
            except ValueError:
                pass
        if json_str:
            try:
                parsed_options.append((datetime.fromisoformat(json_str), "json"))
            except ValueError:
                pass
        if filename_str:
            try:
                parsed_options.append((datetime.fromisoformat(filename_str), "filename"))
            except ValueError:
                pass

        if not parsed_options:
            # no valid date found; leave chosen_date NULL
            continue

        # choose the oldest date (keep source)
        chosen_datetime, chosen_source = min(parsed_options, key=lambda x: x[0])

        # If the winner came from FILENAME and is exactly midnight (00:00:00),
        # prefer any alternative source (EXIF/JSON) if available—pick the oldest
        # among those non-filename options.
        if (
            chosen_source == "filename"
            and chosen_datetime.hour == 0
            and chosen_datetime.minute == 0
            and chosen_datetime.second == 0
        ):
            non_filename_options = [(dt, src) for (dt, src) in parsed_options if src != "filename"]
            if non_filename_options:
                chosen_datetime, _ = min(non_filename_options, key=lambda x: x[0])
        chosen_string = chosen_datetime.strftime("%Y-%m-%d_%H-%M-%S")

        # update the database with the chosen date
        update_chosen_date(connection, Path(filepath), chosen_string)

    connection.commit()