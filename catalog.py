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

Utilities:
- scan_schema(conn) -> dict: returns expected/existing/missing columns and indexes.
- health_check(conn, auto_fix: bool = False) -> str: human-readable summary; can optionally auto-create missing pieces.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable, Tuple, List, Optional
from pathlib import Path
import logging

# Centralized logs directory at project root
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"{Path(__file__).stem}.log"

# Configure per-module file logger (idempotent) to only log errors and above
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # only log errors and above
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(LOG_FILE) for h in logger.handlers):
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setLevel(logging.ERROR)  # ensure handler also filters to errors
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


# -------------------------------
# SQLite performance pragmas
# -------------------------------
def apply_performance_pragmas(connection: sqlite3.Connection) -> None:
    """
    Safe defaults: WAL + NORMAL sync + in-memory temp store + larger cache.
    Call once per connection; idempotent.
    """
    cur = connection.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-65536;")  # ~64MB
    connection.commit()


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
COL_VIDEO_HASH = "video_hash"
COL_SSIM = "ssim_score"
COL_CHOSEN = "chosen_timestamp"
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
    COL_VIDEO_HASH: "TEXT",
    COL_SSIM: "REAL",
    COL_CHOSEN: "TEXT",
    COL_RENAMED: "TEXT",
}

# Common extension sets to help filter by media type without external libs
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif", ".bmp", ".tiff")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".3gp", ".mpg", ".m4v", ".webm", ".mts", ".m2ts", ".mkv")

def _ext_like_clause(column: str, exts: Iterable[str]) -> Tuple[str, Tuple[str, ...]]:
    """
    Build a SQL 'LOWER(column) LIKE ...' OR clause and its parameters for a set of extensions.
    Returns (clause_sql, params_tuple).
    """
    parts = [f"LOWER({column}) LIKE ?"] * len(tuple(exts))
    clause = "(" + " OR ".join(parts) + ")"
    params = tuple(f"%{ext.lower()}" for ext in exts)
    return clause, params


def initialize_database(database_path: Path = DATABASE_FILE) -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database, ensure schema, return connection."""
    connection = sqlite3.connect(str(database_path))
    apply_performance_pragmas(connection)
    ensure_schema(connection)
    ensure_indexes(connection)
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

    # Back-compat: if old column exists, ensure new one and migrate values
    if "chosen_date" in existing and COL_CHOSEN not in existing:
        cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COL_CHOSEN} TEXT")
        cur.execute(
            f"UPDATE {TABLE_NAME} SET {COL_CHOSEN} = chosen_date WHERE {COL_CHOSEN} IS NULL"
        )
        existing.add(COL_CHOSEN)


    connection.commit()


# -------------------------------
# Index management
# -------------------------------
def ensure_indexes(connection: sqlite3.Connection, phash_prefix_len: int = 4) -> None:
    cur = connection.cursor()
    # Consistent, idempotent index names using constants
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_phash ON {TABLE_NAME}({COL_PHASH})")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_dhash ON {TABLE_NAME}({COL_DHASH})")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_video_hash ON {TABLE_NAME}({COL_VIDEO_HASH})")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_chosen_ts ON {TABLE_NAME}({COL_CHOSEN})")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_phash_prefix ON {TABLE_NAME}(substr({COL_PHASH},1,{phash_prefix_len}))")
    connection.commit()

    
# -------------------------------
# Schema/introspection utilities
# -------------------------------
# Required index names, kept consistent with ensure_indexes()
REQUIRED_INDEXES = {
    f"idx_{TABLE_NAME}_phash",
    f"idx_{TABLE_NAME}_dhash",
    f"idx_{TABLE_NAME}_video_hash",
    f"idx_{TABLE_NAME}_chosen_ts",
    f"idx_{TABLE_NAME}_phash_prefix",
}

def _existing_columns(connection: sqlite3.Connection) -> List[str]:
    """Return a list of existing column names on the media table."""
    cur = connection.cursor()
    cur.execute(f"PRAGMA table_info({TABLE_NAME})")
    return [row[1] for row in cur.fetchall()]

def _existing_index_names(connection: sqlite3.Connection) -> List[str]:
    """Return a list of index names currently present on the media table."""
    cur = connection.cursor()
    cur.execute(f"PRAGMA index_list({TABLE_NAME})")
    return [row[1] for row in cur.fetchall()]  # [ (seq, name, unique, origin, partial) ... ]

def scan_schema(connection: sqlite3.Connection) -> dict:
    """
    Scan the catalog for schema consistency.

    Returns:
        {
          "table": <TABLE_NAME>,
          "columns": {
              "expected": [...],
              "existing": [...],
              "missing":  [...],
          },
          "indexes": {
              "expected": [...],
              "existing": [...],
              "missing":  [...],
          }
        }
    """
    existing_cols = set(_existing_columns(connection))
    existing_idx  = set(_existing_index_names(connection))

    expected_cols = set(REQUIRED_COLUMNS.keys()) | {COL_ID, COL_FILEPATH}
    expected_idx  = set(REQUIRED_INDEXES)

    return {
        "table": TABLE_NAME,
        "columns": {
            "expected": sorted(expected_cols),
            "existing": sorted(existing_cols),
            "missing":  sorted(expected_cols - existing_cols),
        },
        "indexes": {
            "expected": sorted(expected_idx),
            "existing": sorted(existing_idx),
            "missing":  sorted(expected_idx - existing_idx),
        },
    }

def health_check(connection: sqlite3.Connection, auto_fix: bool = False) -> str:
    """
    Validate that the database schema and indexes match expectations.

    Args:
        connection: Open sqlite3.Connection
        auto_fix: If True, will create any missing columns/indexes by calling
                  ensure_schema() and ensure_indexes() before re-scanning.

    Returns:
        A human-readable multi-line report summarizing the current status.
    """
    if auto_fix:
        # Best-effort bring-up before scanning for final state
        ensure_schema(connection)
        ensure_indexes(connection)

    report = scan_schema(connection)

    lines: List[str] = []
    lines.append(f"Catalog health check for table '{report['table']}':")

    # Columns
    missing_cols = report["columns"]["missing"]
    if missing_cols:
        lines.append(f"  ❌ Missing column(s): {', '.join(missing_cols)}")
    else:
        lines.append("  ✅ All expected columns are present.")

    # Indexes
    missing_idx = report["indexes"]["missing"]
    if missing_idx:
        lines.append(f"  ❌ Missing index(es): {', '.join(missing_idx)}")
    else:
        lines.append("  ✅ All expected indexes are present.")

    return "\n".join(lines)
    
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


# --- Internal helpers ---

def _update_by_any_path(connection: sqlite3.Connection, set_expr: str, value: object, media_file_path: Path) -> bool:
    """Update a single column using either filepath or renamed_filepath as key."""
    cur = connection.cursor()
    cur.execute(
        f"UPDATE {TABLE_NAME} SET {set_expr} "
        f"WHERE {COL_FILEPATH} = ? OR {COL_RENAMED} = ?",
        (value, str(media_file_path), str(media_file_path)),
    )
    connection.commit()
    return cur.rowcount > 0


# --- Field update helpers ---

def update_json_path(connection: sqlite3.Connection, media_file_path: Path, json_file_path: Path) -> bool:
    """Record the sidecar JSON path for a given media file."""
    return _update_by_any_path(connection, f"{COL_JSONPATH} = ?", str(json_file_path), media_file_path)


def update_exif_create_date(connection: sqlite3.Connection, media_file_path: Path, create_date: str) -> bool:
    """Store the EXIF CreateDate (camera’s recorded timestamp)."""
    return _update_by_any_path(connection, f"{COL_EXIF_CREATE} = ?", create_date, media_file_path)


def update_json_taken_date(connection: sqlite3.Connection, media_file_path: Path, json_taken_date: str) -> bool:
    """Store the Google JSON photoTakenTime timestamp."""
    return _update_by_any_path(connection, f"{COL_JSON_TAKEN} = ?", json_taken_date, media_file_path)


def update_filename_parsed_date(
    connection: sqlite3.Connection,
    media_file_path: Path,
    parsed_date: str,
) -> bool:
    """Store the fallback date parsed from the filename."""
    return _update_by_any_path(connection, f"{COL_FILENAME_PARSED} = ?", parsed_date, media_file_path)


def update_chosen_timestamp(connection: sqlite3.Connection, media_file_path: Path, chosen_string: str) -> bool:
    logger.debug(f"Updating chosen_timestamp for {media_file_path}: {chosen_string}")
    return _update_by_any_path(connection, f"{COL_CHOSEN} = ?", chosen_string, media_file_path)


def update_chosen_date(connection: sqlite3.Connection, media_file_path: Path, chosen_date: str) -> bool:
    """Record the final chosen date string for this file (already formatted)."""
    return _update_by_any_path(connection, f"{COL_CHOSEN} = ?", chosen_date, media_file_path)


# --- Perceptual hash and SSIM update helpers ---

def update_phash(connection: sqlite3.Connection, media_file_path: Path, perceptual_hash: str) -> bool:
    """Store the perceptual pHash for a given media file."""
    return _update_by_any_path(connection, f"{COL_PHASH} = ?", perceptual_hash, media_file_path)


def update_dhash(connection: sqlite3.Connection, media_file_path: Path, difference_hash: str) -> bool:
    """Store the perceptual dHash for a given media file."""
    return _update_by_any_path(connection, f"{COL_DHASH} = ?", difference_hash, media_file_path)


def update_video_hash(connection: sqlite3.Connection, media_file_path: Path, video_hash: str) -> bool:
    """Store the perceptual video hash for a given media file."""
    return _update_by_any_path(connection, f"{COL_VIDEO_HASH} = ?", video_hash, media_file_path)


# -------------------------------
# Batch write helpers
# -------------------------------
def batch_update_ssim(connection: sqlite3.Connection, rows: Iterable[Tuple[float, str]]) -> int:
    """
    Batch update SSIM scores.
    rows: iterable of (ssim_score, any_path) where any_path matches either filepath or renamed_filepath.
    """
    cur = connection.cursor()
    cur.executemany(
        f"UPDATE {TABLE_NAME} SET {COL_SSIM}=? WHERE {COL_FILEPATH}=? OR {COL_RENAMED}=?",
        ((score, fp, fp) for score, fp in rows)
    )
    connection.commit()
    return cur.rowcount or 0


def batch_update_renamed_filepath(connection: sqlite3.Connection, rows: Iterable[Tuple[str, str]]) -> int:
    """
    Batch update renamed paths.
    rows: iterable of (new_filepath, old_filepath) — the old path may be recorded in either filepath or renamed_filepath.
    """
    cur = connection.cursor()
    cur.executemany(
        f"UPDATE {TABLE_NAME} SET {COL_RENAMED}=? WHERE {COL_RENAMED}=? OR {COL_FILEPATH}=?",
        ((new_fp, old_fp, old_fp) for new_fp, old_fp in rows)
    )
    connection.commit()
    return cur.rowcount or 0


# --- Queries ---

def fetch_unhashed_images(connection: sqlite3.Connection, limit: Optional[int] = None) -> List[str]:
    """
    Return filepaths that look like images and do not yet have pHash/dHash.
    """
    clause, params = _ext_like_clause(COL_FILEPATH, IMAGE_EXTS)
    sql = (
        f"SELECT {COL_FILEPATH} FROM {TABLE_NAME} "
        f"WHERE {COL_PHASH} IS NULL AND {COL_DHASH} IS NULL "
        f"AND {clause} "
        f"ORDER BY {COL_ID} ASC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params = params + (int(limit),)
    cur = connection.cursor()
    cur.execute(sql, params)
    return [row[0] for row in cur.fetchall()]

def fetch_unhashed_videos(connection: sqlite3.Connection, limit: Optional[int] = None) -> List[str]:
    """
    Return filepaths that look like videos and do not yet have a video hash.
    """
    clause, params = _ext_like_clause(COL_FILEPATH, VIDEO_EXTS)
    sql = (
        f"SELECT {COL_FILEPATH} FROM {TABLE_NAME} "
        f"WHERE {COL_VIDEO_HASH} IS NULL "
        f"AND {clause} "
        f"ORDER BY {COL_ID} ASC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params = params + (int(limit),)
    cur = connection.cursor()
    cur.execute(sql, params)
    return [row[0] for row in cur.fetchall()]

def fetch_without_json(connection: sqlite3.Connection, limit: Optional[int] = None) -> List[str]:
    """
    Return filepaths with no associated JSON sidecar path recorded.
    """
    sql = (
        f"SELECT {COL_FILEPATH} FROM {TABLE_NAME} "
        f"WHERE {COL_JSONPATH} IS NULL "
        f"ORDER BY {COL_ID} ASC"
    )
    params: Tuple[object, ...] = tuple()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    cur = connection.cursor()
    cur.execute(sql, params)
    return [row[0] for row in cur.fetchall()]

def fetch_missing_timestamps(connection: sqlite3.Connection, limit: Optional[int] = None) -> List[str]:
    """
    Return filepaths where the final chosen timestamp is still missing.
    """
    sql = (
        f"SELECT {COL_FILEPATH} FROM {TABLE_NAME} "
        f"WHERE {COL_CHOSEN} IS NULL "
        f"ORDER BY {COL_ID} ASC"
    )
    params: Tuple[object, ...] = tuple()
    if limit is not None:
        sql += " LIMIT ?"
        params = (int(limit),)
    cur = connection.cursor()
    cur.execute(sql, params)
    return [row[0] for row in cur.fetchall()]

def fetch_all_media_entries(connection: sqlite3.Connection) -> List[Tuple[str, ...]]:
    """Return all media rows with commonly used fields."""
    cur = connection.cursor()
    cur.execute(
        f"""
        SELECT {COL_FILEPATH}, {COL_JSONPATH}, {COL_EXIF_CREATE},
               {COL_JSON_TAKEN}, {COL_FILENAME_PARSED},
               {COL_PHASH}, {COL_DHASH}, {COL_VIDEO_HASH}, {COL_SSIM},
               {COL_CHOSEN}, {COL_RENAMED}
        FROM {TABLE_NAME}
        """
    )
    return cur.fetchall()

def vacuum_and_checkpoint(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    connection.execute("VACUUM;")


# CLI health check entry point
if __name__ == "__main__":
    # Allow quick CLI health check: `python catalog.py`
    conn = initialize_database(DATABASE_FILE)
    print(health_check(conn, auto_fix=False))