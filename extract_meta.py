#!/usr/bin/env python3
"""
extract_meta.py

Provides functions to discover media and JSON sidecar files, match JSON to media
by internal title, and extract timestamps from EXIF (CreateDate) and JSON (photoTakenTime).
Also includes a filename-based fallback date parser.
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor

# Ensure a logs directory exists next to this script
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = LOGS_DIR / "extract_meta.log"
handler = logging.FileHandler(log_file, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# Supported media file extensions
MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic",
    ".mp4", ".mov", ".avi", ".3gp", ".mpg", ".m4v", ".gif",
}

# Mapping from MIME types to preferred file extensions
MIME_EXTENSION_MAP = {
    "image/jpeg":   ".jpg",
    "image/png":    ".png",
    "image/webp":   ".webp",
    "image/heic":   ".heic",
    "image/gif":    ".gif",
    "video/mp4":    ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/3gpp":   ".3gp",
    "video/mpeg":   ".mpg",
    "video/x-m4v":  ".m4v",
}

# Processed/errored JSON folder names
PROCESSED_JSON_DIR_NAME = "json_processed"
ERROR_JSON_DIR_NAME = "json_error"

# JSON sidecar filename suffixes to ignore
IGNORE_JSON_SUFFIXES = {".db", ".txt", ".md"}

# Regex for parsing dates out of filenames, e.g. IMG_20220101_123456
FILENAME_DATE_PATTERN = re.compile(
    r"""
    (?P<year>(?:19|20)\d{2})           # 4-digit year starting with 19xx or 20xx
    [-_.]?                              # optional separator
    (?P<month>\d{2})                   # 2-digit month
    [-_.]?                              # optional separator
    (?P<day>\d{2})                     # 2-digit day
    (?:                                 # --- optional time portion ---
        [ T_-]?                         # optional spacer between date and time
        (?:
            # Separated form requires at least HH:MM
            (?P<hour_sep>\d{2})(?:(?::|\.)(?P<minute_sep>\d{2})(?:(?::|\.)(?P<second_sep>\d{2}))?)
            |
            # Compact form must be HHMM or HHMMSS (not 2 or 3 digits)
            (?P<hour_c>\d{2})(?P<minute_c>\d{2})(?P<second_c>\d{2})?
        )
    )?
    """,
    re.VERBOSE,
)


def find_media_files(root_directory: Path) -> List[Path]:
    """
    Recursively find all media files under root_directory matching supported extensions.
    """
    root_directory = Path(root_directory)
    media_file_paths: List[Path] = []
    for candidate in root_directory.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in MEDIA_EXTENSIONS:
            media_file_paths.append(candidate)
    return media_file_paths


def find_json_files(root_directory: Path) -> List[Path]:
    """
    Recursively find all JSON sidecar files under root_directory (ending in .json).
    """
    root_directory = Path(root_directory)
    json_file_paths: List[Path] = []
    for candidate in root_directory.rglob("*.json"):
        if candidate.suffix.lower() != ".json":
            continue
        if candidate.name.lower().endswith(tuple(IGNORE_JSON_SUFFIXES)):
            continue
        json_file_paths.append(candidate)
    return json_file_paths


def build_json_title_index(json_file_paths: List[Path]) -> dict:
    """
    Build a mapping from JSON internal title stem (lowercase) to JSON path.
    """
    title_map = {}
    for json_path in json_file_paths:
        try:
            sidecar = json.loads(json_path.read_text(encoding="utf-8"))
            raw_title = sidecar.get("title", "")
            clean_title = raw_title.replace("\\", "")  # unescape backslashes
            stem = Path(clean_title).stem.lower()
            if stem:
                title_map[stem] = json_path
        except Exception:
            continue
    return title_map

# In-memory cache so we don't re-read every JSON for every media match
_CACHED_JSON_INDEX: Dict[str, list[Path]] | None = None
_CACHED_JSON_SET: set[str] | None = None


def _normalize_title_to_stems(title: str) -> list[str]:
    clean = title.replace("\\", "")
    stem = Path(clean).stem.lower()
    if not stem:
        return []
    stems = {stem}
    stems.add(re.sub(r"\(\d+\)$", "", stem))  # remove (n)
    if stem.endswith("-edited"):
        stems.add(stem[: -len("-edited")])
    if stem.endswith("ani"):
        stems.add(stem[:-1])  # '-ani' -> '-an'
    return [s for s in stems if s]


def build_json_title_index(json_file_paths: List[Path]) -> Dict[str, list[Path]]:
    def parse_one(jpath: Path) -> tuple[list[str], Path] | None:
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            stems = _normalize_title_to_stems(data.get("title", ""))
            if stems:
                return stems, jpath
        except Exception:
            return None
        return None

    index: Dict[str, list[Path]] = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        for result in tqdm(ex.map(parse_one, json_file_paths), total=len(json_file_paths),
                           desc="🧩 Indexing JSON titles", unit="file"):
            if not result:
                continue
            stems, jpath = result
            for s in stems:
                index.setdefault(s, []).append(jpath)
    return index


def _ensure_json_index(json_file_paths: List[Path]) -> Dict[str, list[Path]]:
    global _CACHED_JSON_INDEX, _CACHED_JSON_SET
    current_set = {str(p) for p in json_file_paths}
    if _CACHED_JSON_INDEX is None or _CACHED_JSON_SET != current_set:
        _CACHED_JSON_INDEX = build_json_title_index(json_file_paths)
        _CACHED_JSON_SET = current_set
    return _CACHED_JSON_INDEX


def match_json_from_index(media_file_path: Path, title_index: Dict[str, list[Path]]) -> Optional[Path]:
    """
    Match a media file to its JSON sidecar using a prebuilt title index.
    """
    media_stem = media_file_path.stem.lower()
    candidates = (
        title_index.get(media_stem)
        or title_index.get(re.sub(r"\(\d+\)$", "", media_stem))
        or title_index.get(media_stem.rstrip("i"))  # '-ani' -> '-an'
    )
    if not candidates:
        return None
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def match_json_for_media(media_file_path: Path, json_file_paths: List[Path]) -> Optional[Path]:
    title_index = _ensure_json_index(json_file_paths)
    return match_json_from_index(media_file_path, title_index)


def extract_exif_create_date(media_file_path: Path) -> Optional[str]:
    """
    Use ExifTool to read the EXIF CreateDate tag and return it in UTC ISO format.
    Returns None if not found or on error.
    """
    logger.debug(f"Extracting EXIF CreateDate for {media_file_path}")

    for tag in ("CreateDate", "DateTimeOriginal", "ModifyDate"):
        try:
            output = subprocess.check_output(
                [
                    "exiftool", "-s3", f"-{tag}", str(media_file_path)
                ], stderr=subprocess.DEVNULL
            ).decode().strip()
            if not output:
                continue
            # Parse "YYYY:MM:DD HH:MM:SS" into datetime
            naive_datetime = datetime.strptime(output, "%Y:%m:%d %H:%M:%S")
            # Assume local time of original; convert to UTC
            aware_datetime = naive_datetime.replace(tzinfo=timezone.utc)
            return aware_datetime.isoformat()
        except Exception:
            continue
    logger.warning(f"No EXIF CreateDate/DateTimeOriginal/ModifyDate found for {media_file_path}")
    return None


def extract_json_taken_date(media_file_path: Path, json_file_paths: List[Path], target_root: Path) -> Optional[str]:
    """
    Read JSON sidecar's photoTakenTime or creationTime timestamp by matching
    via internal title map. Returns UTC ISO string or None.

    Sidecars are always moved under the provided `target_root`:
      - <target_root>/json_processed on success
      - <target_root>/json_error on failure
    """
    # Find the correct JSON sidecar
    matched_json = match_json_for_media(media_file_path, json_file_paths)
    if not matched_json:
        logger.warning(f"No JSON sidecar matched for {media_file_path}")
        return None

    json_path = Path(matched_json)
    processed_dir = target_root / PROCESSED_JSON_DIR_NAME
    error_dir = target_root / ERROR_JSON_DIR_NAME
    processed_dir.mkdir(exist_ok=True)
    error_dir.mkdir(exist_ok=True)

    try:
        sidecar = json.loads(json_path.read_text(encoding="utf-8"))
        timestamp = (
            sidecar.get("photoTakenTime", {}).get("timestamp")
            or sidecar.get("creationTime", {}).get("timestamp")
        )
        if not timestamp:
            logger.warning(f"No timestamp field in JSON sidecar {json_path}")
            raise ValueError("No timestamp in JSON sidecar")

        # Move JSON to processed
        destination = processed_dir / json_path.name
        try:
            json_path.rename(destination)
        except Exception:
            # If cross-device rename fails, fall back to copy+remove
            try:
                destination.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
                json_path.unlink(missing_ok=True)
            except Exception:
                logger.exception(f"Failed to move JSON to processed: {json_path} -> {destination}")
        logger.info(f"Moved JSON to processed: {destination}")

        taken_dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        return taken_dt.isoformat()

    except Exception:
        # Move JSON to error
        destination = error_dir / json_path.name
        try:
            json_path.rename(destination)
        except Exception:
            try:
                destination.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
                json_path.unlink(missing_ok=True)
            except Exception:
                logger.exception(f"Failed to move JSON to error: {json_path} -> {destination}")
        logger.info(f"Moved JSON to error: {destination}")
        return None


def parse_date_from_filename(filename_stem: str) -> Optional[str]:
    """
    Parse a fallback date from filename patterns like YYYYMMDD_HHMMSS.
    Returns an ISO8601 date string or None.
    """

    match = FILENAME_DATE_PATTERN.search(filename_stem)
    if not match:
        logger.warning(f"No date pattern found in filename: {filename_stem}")
        return None
    
    parts = match.groupdict()
    try:
        year = int(parts["year"])  
        month = int(parts["month"]) 
        day = int(parts["day"])    

        hour = 0
        minute = 0
        second = 0

        if parts.get("hour_sep"):
            # Separated time: HH:MM[:SS]
            hour = int(parts["hour_sep"])  
            minute = int(parts.get("minute_sep") or 0)
            second = int(parts.get("second_sep") or 0)
        elif parts.get("hour_c") and parts.get("minute_c"):
            # Compact time: HHMM[SS]
            hour = int(parts["hour_c"])  
            minute = int(parts["minute_c"]) 
            second = int(parts.get("second_c") or 0)

        parsed_dt = datetime(
            year, month, day,
            hour, minute, second,
            tzinfo=timezone.utc,
        )
        return parsed_dt.isoformat()
    except Exception:
        logger.exception(f"Failed to parse date from filename: {filename_stem}")


# Detect and correct file extension based on MIME type
def correct_file_extension_by_mime(media_file_path: Path) -> Path:
    """
    Detect the file's MIME type via `file --mime-type`, and rename it
    on disk if its extension does not match the preferred extension.
    Returns the (possibly renamed) Path.
    """
    try:
        # Get the mime type
        mime = subprocess.check_output(
            ["file", "--mime-type", "-b", str(media_file_path)],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        preferred_ext = MIME_EXTENSION_MAP.get(mime)
        if preferred_ext and media_file_path.suffix.lower() != preferred_ext:
            new_path = media_file_path.with_suffix(preferred_ext)
            media_file_path.rename(new_path)
            return new_path
    except Exception:
        pass
    return media_file_path
