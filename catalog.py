#!/usr/bin/env python3
"""
catalog.py

Manages the media catalog database for the Takeout Organizer pipeline,
storing paths and timestamps from JSON, EXIF, filename parsing, and
the final chosen datetime.
"""

import sqlite3
from pathlib import Path

# Default database file name (in your working directory)
DATABASE_FILE = Path("media_catalog.db")


def initialize_database(database_path: Path = DATABASE_FILE) -> sqlite3.Connection:
    """
    Create (if needed) and open the SQLite database.
    Returns a Connection object.
    """
    connection = sqlite3.connect(str(database_path))
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY,
            filepath TEXT UNIQUE NOT NULL,
            jsonpath TEXT,
            exif_modify_date TEXT,
            json_taken_date TEXT,
            filename_parsed_date TEXT,
            chosen_date TEXT
        )
        """
    )
    connection.commit()
    return connection


def add_media_entry(connection: sqlite3.Connection, media_file_path: Path) -> None:
    """
    Ensure there is a row for this media file (no-op if it already exists).
    """
    cursor = connection.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO media (filepath) VALUES (?)",
        (str(media_file_path),)
    )
    connection.commit()


def update_json_path(connection: sqlite3.Connection, media_file_path: Path, json_file_path: Path) -> None:
    """
    Record the side-car JSON path for a given media file.
    """
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE media SET jsonpath = ? WHERE filepath = ?",
        (str(json_file_path), str(media_file_path))
    )
    connection.commit()


def update_exif_modify_date(connection: sqlite3.Connection, media_file_path: Path, modify_date: str) -> None:
    """
    Store the EXIF ModifyDate (the camera’s recorded timestamp).
    """
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE media SET exif_modify_date = ? WHERE filepath = ?",
        (modify_date, str(media_file_path))
    )
    connection.commit()


def update_json_taken_date(connection: sqlite3.Connection, media_file_path: Path, json_taken_date: str) -> None:
    """
    Store the Google JSON photoTakenTime timestamp.
    """
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE media SET json_taken_date = ? WHERE filepath = ?",
        (json_taken_date, str(media_file_path))
    )
    connection.commit()


def update_filename_parsed_date(
    connection: sqlite3.Connection,
    media_file_path: Path,
    parsed_date: str
) -> None:
    """
    Store the fallback date parsed from the filename.
    """
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE media SET filename_parsed_date = ? WHERE filepath = ?",
        (parsed_date, str(media_file_path))
    )
    connection.commit()


def update_chosen_date(connection: sqlite3.Connection, media_file_path: Path, chosen_date: str) -> None:
    """
    Record the final chosen UTC date for this file.
    """
    cursor = connection.cursor()
    cursor.execute(
        "UPDATE media SET chosen_date = ? WHERE filepath = ?",
        (chosen_date, str(media_file_path))
    )
    connection.commit()


def fetch_all_media_entries(connection: sqlite3.Connection) -> list[tuple]:
    """
    Return a list of all rows with:
    (filepath, jsonpath, exif_modify_date, json_taken_date,
     filename_parsed_date, chosen_date)
    """
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT filepath, jsonpath, exif_modify_date,
               json_taken_date, filename_parsed_date, chosen_date
        FROM media
        """
    )
    return cursor.fetchall()