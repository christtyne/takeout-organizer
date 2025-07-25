

#!/usr/bin/env python3
"""
choose_timestamp.py

Choose the final 'chosen_date' for each media entry in the SQLite catalog
based on priority:
  1. EXIF ModifyDate (most reliable)
  2. JSON photoTakenTime (fallback)
  3. Filename-parsed date
"""

import sqlite3
from datetime import datetime
from tqdm import tqdm

def choose_timestamp_for_all(connection: sqlite3.Connection) -> None:
    """
    For each media record, select the oldest available timestamp among
    exif_modify_date, json_taken_date, and filename_parsed_date, then
    update the chosen_date field.
    """
    cursor = connection.cursor()
    cursor.execute("""
        SELECT filepath, exif_modify_date, json_taken_date, filename_parsed_date
        FROM media
    """)
    rows = cursor.fetchall()

    for filepath, exif_str, json_str, filename_str in tqdm(rows, desc="Choosing timestamps", unit="file"):
        # Collect valid datetime objects
        parsed_options = []
        if exif_str:
            try:
                parsed_options.append(datetime.fromisoformat(exif_str))
            except ValueError:
                pass
        if json_str:
            try:
                parsed_options.append(datetime.fromisoformat(json_str))
            except ValueError:
                pass
        if filename_str:
            try:
                parsed_options.append(datetime.fromisoformat(filename_str))
            except ValueError:
                pass

        if not parsed_options:
            # no valid date found; leave chosen_date NULL
            continue

        # choose the oldest (earliest) date
        chosen_datetime = min(parsed_options)
        chosen_string = chosen_datetime.isoformat()

        # update the database with the chosen date
        cursor.execute(
            "UPDATE media SET chosen_date = ? WHERE filepath = ?",
            (chosen_string, filepath)
        )

    connection.commit()