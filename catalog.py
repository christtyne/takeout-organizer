#!/usr/bin/env python3
"""
catalog.py

Manage the media catalog database for the Takeout Organizer pipeline.
This module centralizes all DB I/O and enforces a stable schema.

Key improvements over previous revisions:
- Robust schema management via ensure_schema(): adds any missing columns.
- Consistent type hints and boolean return values for update helpers.
- Robust path-based updates that match by filepath OR renamed_filepath.
- Idempotent SSIM updates that preserve the maximum score seen.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

# Default database file name (in your working directory)
DATABASE_FILE = Path("media_catalog.db")

# --- Schema definition ---
TABLE_NAME = "media"

# Column names (use constants to avoid typos)
COL_ID = "id"
COL_FILEPATH = "filepath"
COL_JSONPATH = "jsonpath"
COL_EXIF_CREATE = "exif_create_date"
COL_JSON_TAKEN = "json_taken_date"
COL_FILENAME_PARSED = "filename_parsed_date"
COL_PHASH = "phash"
COL_DHASH = "dhash"
COL_SSIM = "ssim_score"
COL_CHOSEN = "chosen_date"
COL_RENAMED = "renamed_filepath"

# Required columns and SQLite types
REQUIRED_COLUMNS = {
    COL_FILEPATH: "TEXT UNIQUE NOT NULL",
    COL_JSONPATH: "TEXT",
    COL_EXIF_CREATE: "TEXT",
    COL_JSON_TAKEN: "TEXT",
    COL_FILENAME_PARSED: "TEXT",
    COL_PHASH: "TEXT",
    COL_DHASH: "TEXT",
    COL_SSIM: "REAL",
    COL_CHOSEN: "TEXT",
    COL_RENAMED: "TEXT",
}


def initialize_database(database_path: Path = DATABASE_FILE) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database, ensure schema, return connection."""
    connection = sqlite3.connect(str(database_path))
    ensure_schema(connection)
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Ensure the `media` table exists and contains all required columns.

    This function is safe to call repeatedly; it will add any missing columns
    without damaging existing data.
    """
    cur = connection.cursor()

    # Create table if missing with minimal shape
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            {COL_ID} INTEGER PRIMARY KEY,
            {COL_FILEPATH} TEXT UNIQUE NOT NULL
        )
        """
    )

    # Discover existing columns
    cur.execute(f"PRAGMA table_info({TABLE_NAME})")
    existing = {row[1] for row in cur.fetchall()}  # column names

    # Add any missing columns
    for col_name, col_decl in REQUIRED_COLUMNS.items():
        if col_name not in existing:
            cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col_name} {col_decl}")

    connection.commit()


# --- Row creation ---

def add_media_entry(connection: sqlite3.Connection, media_file_path: Path) -> bool:
    """Ensure there is a row for this media file. Returns True if inserted."""
    cur = connection.cursor()
    cur.execute(
        f"INSERT OR IGNORE INTO {TABLE_NAME} ({COL_FILEPATH}) VALUES (?)",
        (str(media_file_path),),
    )
    connection.commit()
    return cur.rowcount > 0


# --- Field update helpers ---

def update_json_path(connection: sqlite3.Connection, media_file_path: Path, json_file_path: Path) -> bool:
    """Record the sidecar JSON path for a given media file."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {COL_JSONPATH} = ? WHERE {COL_FILEPATH} = ?",
        (str(json_file_path), str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


def update_exif_create_date(connection: sqlite3.Connection, media_file_path: Path, create_date: str) -> bool:
    """Store the EXIF CreateDate (camera’s recorded timestamp)."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {COL_EXIF_CREATE} = ? WHERE {COL_FILEPATH} = ?",
        (create_date, str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


def update_json_taken_date(connection: sqlite3.Connection, media_file_path: Path, json_taken_date: str) -> bool:
    """Store the Google JSON photoTakenTime timestamp."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {COL_JSON_TAKEN} = ? WHERE {COL_FILEPATH} = ?",
        (json_taken_date, str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


def update_filename_parsed_date(
    connection: sqlite3.Connection,
    media_file_path: Path,
    parsed_date: str,
) -> bool:
    """Store the fallback date parsed from the filename."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {COL_FILENAME_PARSED} = ? WHERE {COL_FILEPATH} = ?",
        (parsed_date, str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


def update_chosen_date(connection: sqlite3.Connection, media_file_path: Path, chosen_date: str) -> bool:
    """Record the final chosen date string for this file (already formatted)."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {COL_CHOSEN} = ? WHERE {COL_FILEPATH} = ?",
        (chosen_date, str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


# --- Perceptual hash and SSIM update helpers ---

def update_phash(connection: sqlite3.Connection, media_file_path: Path, perceptual_hash: str) -> bool:
    """Store the perceptual pHash for a given media file."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {COL_PHASH} = ? WHERE {COL_FILEPATH} = ?",
        (perceptual_hash, str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


def update_dhash(connection: sqlite3.Connection, media_file_path: Path, difference_hash: str) -> bool:
    """Store the perceptual dHash for a given media file."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {COL_DHASH} = ? WHERE {COL_FILEPATH} = ?",
        (difference_hash, str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


def update_ssim_score(connection: sqlite3.Connection, media_file_path: Path, score: float) -> bool:
    """Store SSIM for a given file, preserving the maximum score seen.

    - Casts to built-in float to avoid numpy types.
    - Matches by current filepath OR renamed_filepath.
    Returns True if a row was updated.
    """
    cur = connection.cursor()
    val = float(score)
    cur.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET {COL_SSIM} = CASE
            WHEN {COL_SSIM} IS NULL OR ? > {COL_SSIM} THEN ?
            ELSE {COL_SSIM}
        END
        WHERE {COL_FILEPATH} = ? OR {COL_RENAMED} = ?
        """,
        (val, val, str(media_file_path), str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


def update_renamed_filepath(connection: sqlite3.Connection, old_file_path: Path, new_file_path: Path) -> bool:
    """Update renamed_filepath to `new_file_path` matching by either filepath or renamed_filepath."""
    cur = connection.cursor()
    cur.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET {COL_RENAMED} = ?
        WHERE {COL_FILEPATH} = ? OR {COL_RENAMED} = ?
        """,
        (str(new_file_path), str(old_file_path), str(old_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


# --- Queries ---

def fetch_all_media_entries(connection: sqlite3.Connection) -> List[Tuple[str, ...]]:
    """Return all media rows with commonly used fields."""
    cur = connection.cursor()
    cur.execute(
        f"""
        SELECT {COL_FILEPATH}, {COL_JSONPATH}, {COL_EXIF_CREATE},
               {COL_JSON_TAKEN}, {COL_FILENAME_PARSED},
               {COL_PHASH}, {COL_DHASH}, {COL_SSIM},
               {COL_CHOSEN}, {COL_RENAMED}
        FROM {TABLE_NAME}
        """
    )
    return cur.fetchall()