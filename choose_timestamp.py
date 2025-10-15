#!/usr/bin/env python3
"""
choose_timestamp.py

Choose the final 'chosen_timestamp' for each media entry in the SQLite catalog
based on priority:
  1. EXIF CreateDate (most reliable)
  2. JSON photoTakenTime (fallback)
  3. Filename-parsed date
"""

import sqlite3
import re
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import logging
from catalog import update_chosen_timestamp

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

def choose_timestamp_for_all(connection: sqlite3.Connection) -> None:
    """
    For each media record missing a chosen_timestamp, pick the oldest available timestamp among:
      - exif_create_date (most reliable)
      - json_taken_date
      - filename_parsed_date
    and update chosen_timestamp using a single batch write for performance.
    """
    cursor = connection.cursor()

    # Only consider rows that have at least one timestamp source available
    where_clause = """
        chosen_timestamp IS NULL
        AND (exif_create_date IS NOT NULL OR json_taken_date IS NOT NULL OR filename_parsed_date IS NOT NULL)
    """

    # Progress total
    cursor.execute(f"SELECT COUNT(*) FROM media WHERE {where_clause}")
    total = cursor.fetchone()[0] or 0

    # Stream rows in batches to avoid high memory usage on very large catalogs
    cursor.execute(f"""
        SELECT filepath, exif_create_date, json_taken_date, filename_parsed_date
        FROM media
        WHERE {where_clause}
        ORDER BY id ASC
    """)

    def _safe_parse_iso(value: str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None

    def _cmp_key(dt: datetime) -> datetime:
        # Compare aware/naive uniformly without altering the stored value
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    updates = []
    processed = 0
    with tqdm(total=total, desc="🕰️ Choosing timestamps", unit="file", leave=False) as pbar:
        while True:
            rows = cursor.fetchmany(5000)
            if not rows:
                break
            for filepath, exif_str, json_str, filename_str in rows:
                parsed_options = []
                dt_exif = _safe_parse_iso(exif_str) if exif_str else None
                if dt_exif:
                    parsed_options.append((dt_exif, "exif"))
                dt_json = _safe_parse_iso(json_str) if json_str else None
                if dt_json:
                    parsed_options.append((dt_json, "json"))
                dt_filename = _safe_parse_iso(filename_str) if filename_str else None
                if dt_filename:
                    parsed_options.append((dt_filename, "filename"))

                if not parsed_options:
                    logger.error(f"No valid timestamps for {filepath}")
                    pbar.update(1)
                    processed += 1
                    continue

                chosen_datetime, chosen_source = min(parsed_options, key=lambda x: _cmp_key(x[0]))

                # If filename-derived and exactly midnight, prefer any non-filename option (oldest among them)
                if (chosen_source == "filename"
                    and chosen_datetime.hour == 0
                    and chosen_datetime.minute == 0
                    and chosen_datetime.second == 0):
                    non_filename = [(dt, src) for (dt, src) in parsed_options if src != "filename"]
                    if non_filename:
                        chosen_datetime, _ = min(non_filename, key=lambda x: _cmp_key(x[0]))

                chosen_string = chosen_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                updates.append((chosen_string, filepath))
                pbar.update(1)
                processed += 1

    if updates:
        try:
            cursor.executemany("UPDATE media SET chosen_timestamp=? WHERE filepath=?", updates)
            connection.commit()
        except Exception:
            connection.rollback()
            raise