#!/usr/bin/env python3
"""
extract_meta.py

Provides functions to discover media and JSON sidecar files, match JSON to media
by internal title, and extract timestamps from EXIF (CreateDate) and JSON (photoTakenTime).
Also includes a filename-based fallback date parser.
"""

import json
import re
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from PIL import Image
from imagehash import phash, dhash

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

# JSON sidecar filename suffixes to ignore
IGNORE_JSON_SUFFIXES = {".db", ".txt", ".md"}

# Regex for parsing dates out of filenames, e.g. IMG_20220101_123456
FILENAME_DATE_PATTERN = re.compile(
    r"^(?P<year>\d{4})[-_.]?(?P<month>\d{2})[-_.]?(?P<day>\d{2})"
    r"[_-]?(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})$"
)


def find_media_files(root_directory: Path) -> List[Path]:
    """
    Recursively find all media files under root_directory matching supported extensions.
    """
    media_file_paths: List[Path] = []
    for candidate in tqdm(root_directory.rglob("*"), desc="Searching media files", unit="file"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in MEDIA_EXTENSIONS:
            media_file_paths.append(candidate)
    return media_file_paths


def find_json_files(root_directory: Path) -> List[Path]:
    """
    Recursively find all JSON sidecar files under root_directory (ending in .json).
    """
    json_file_paths: List[Path] = []
    for candidate in tqdm(root_directory.rglob("*.json"), desc="Searching JSON files", unit="file"):
        if candidate.suffix.lower() != ".json":
            continue
        if candidate.name.lower().endswith(tuple(IGNORE_JSON_SUFFIXES)):
            continue
        json_file_paths.append(candidate)
    return json_file_paths


def match_json_for_media(media_file_path: Path, json_file_paths: List[Path]) -> Optional[Path]:
    """
    Match a media file to its JSON sidecar by comparing the JSON's internal title field.
    Returns the matching Path or None.
    """
    # Build map: internal title stem (lowercase) -> JSON path
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
    # Derive media filename stem and match
    media_stem = media_file_path.stem.lower()
    # Direct match
    if media_stem in title_map:
        return title_map[media_stem]
    # Strip duplicate indicator (n) e.g. "name(1)"
    stripped_stem = re.sub(r"\(\d+\)$", "", media_stem)
    return title_map.get(stripped_stem)


def extract_exif_create_date(media_file_path: Path) -> Optional[str]:
    """
    Use ExifTool to read the EXIF CreateDate tag and return it in UTC ISO format.
    Returns None if not found or on error.
    """
    try:
        output = subprocess.check_output(
            [
                "exiftool", "-s3", "-CreateDate", str(media_file_path)
            ], stderr=subprocess.DEVNULL
        ).decode().strip()
        if not output:
            return None
        # Parse "YYYY:MM:DD HH:MM:SS" into datetime
        naive_datetime = datetime.strptime(output, "%Y:%m:%d %H:%M:%S")
        # Assume local time of original; convert to UTC
        aware_datetime = naive_datetime.replace(tzinfo=timezone.utc)
        return aware_datetime.isoformat()
    except Exception:
        return None


def extract_json_taken_date(media_file_path: Path, json_file_paths: List[Path]) -> Optional[str]:
    """
    Read JSON sidecar's photoTakenTime or creationTime timestamp by matching
    via internal title map. Returns UTC ISO string or None.
    """
    # Find the correct JSON sidecar
    matched_json = match_json_for_media(media_file_path, json_file_paths)
    if not matched_json:
        return None
    try:
        sidecar = json.loads(Path(matched_json).read_text(encoding="utf-8"))
        timestamp = (
            sidecar.get("photoTakenTime", {}).get("timestamp")
            or sidecar.get("creationTime", {}).get("timestamp")
        )
        if not timestamp:
            return None
        taken_datetime = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
        return taken_datetime.isoformat()
    except Exception:
        return None



def parse_date_from_filename(filename_stem: str) -> Optional[str]:
    """
    Parse a fallback date from filename patterns like YYYYMMDD_HHMMSS.
    Returns an ISO8601 date string or None.
    """
    match = FILENAME_DATE_PATTERN.match(filename_stem)
    if not match:
        return None
    parts = match.groupdict()
    try:
        parsed = datetime(
            int(parts["year"]), int(parts["month"]), int(parts["day"]),
            int(parts["hour"]), int(parts["minute"]), int(parts["second"]),
            tzinfo=timezone.utc
        )
        return parsed.isoformat()
    except Exception:
        return None


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


# Perceptual hash utilities
def compute_perceptual_hashes(media_file_path: Path) -> tuple[str, str]:
    """
    Compute perceptual pHash and dHash for the given media file.
    Returns a tuple (pHash, dHash) as hex strings, or (None, None) on error.
    """
    try:
        image = Image.open(media_file_path).convert("RGB")
        return str(phash(image)), str(dhash(image))
    except Exception:
        return None, None

