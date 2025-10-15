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
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from tqdm import tqdm
import logging
import shutil
import mimetypes

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

@lru_cache(maxsize=2048)
def fast_guess_mime(path: str) -> Optional[str]:
    """Cached best-effort MIME guess using the file extension."""
    try:
        return mimetypes.guess_type(path)[0]
    except Exception:
        return None

from typing import Iterable, Iterator, Any
def _chunked(seq: Iterable[Any], size: int) -> Iterator[list]:
    """Yield lists of up to `size` items from `seq`."""
    batch: list = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

# External tools
EXIFTOOL = "exiftool"
FILE_CMD = "file"

# Reasonable batch size for exiftool to avoid argv limits on large libraries
EXIF_BATCH_SIZE = 500


# Centralized logs directory at project root
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"{Path(__file__).stem}.log"

# Configure per-module file logger (idempotent)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(LOG_FILE) for h in logger.handlers):
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setLevel(logging.ERROR)  # ensure handler filters to errors
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

# Supported media file extensions
MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic",
    ".mp4", ".mov", ".avi", ".3gp", ".mpg", ".m4v", ".gif",
    ".webm", ".mkv", ".ts",
}

# Mapping from MIME types to preferred file extensions
MIME_EXTENSION_MAP = {
    "image/jpeg":   ".jpg",
    "image/png":    ".png",
    "image/webp":   ".webp",
    "image/heic":   ".heic",
    "image/heif":   ".heic",
    "image/heif-sequence": ".heic",
    "image/gif":    ".gif",
    "video/mp4":    ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/3gpp":   ".3gp",
    "video/mpeg":   ".mpg",
    "video/x-m4v":  ".m4v",
    "video/webm":  ".webm",
    "video/x-matroska": ".mkv",
    "video/MP2T":  ".ts",
}

# Processed/errored JSON folder names
PROCESSED_JSON_DIR_NAME = "json_processed"
ERROR_JSON_DIR_NAME = "json_error"

# JSON sidecar filename suffixes to ignore
IGNORE_JSON_SUFFIXES = {".db", ".txt", ".md"}

# Directories/files to skip during scans
SKIP_DIRS = {"__MACOSX", ".Trash", ".Trashes", ".Spotlight-V100"}
SKIP_FILES = {".DS_Store", "Thumbs.db"}

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
    Uses os.walk for lower overhead on very large trees.
    Prunes known noisy/system folders and ignorable files.
    """
    root_directory = Path(root_directory)
    media_file_paths: List[Path] = []
    lower_exts = {e.lower() for e in MEDIA_EXTENSIONS}
    for root, dirs, files in os.walk(root_directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]  # prune
        root_p = Path(root)
        for name in files:
            if name in SKIP_FILES:
                continue
            if Path(name).suffix.lower() in lower_exts:
                media_file_paths.append(root_p / name)
    return media_file_paths


_EXIF_PATTERNS = (
    "%Y:%m:%d %H:%M:%S%z",   # with offset
    "%Y:%m:%d %H:%M:%S",     # classic EXIF without offset
    "%Y-%m-%d %H:%M:%S%z",   # alt separator with offset
    "%Y-%m-%d %H:%M:%S",     # alt separator without offset
)

def _parse_exif_datetime(dt_text: str) -> Optional[datetime]:
    """Parse common EXIF datetime strings, supporting Z and +0200/+02:00 offsets."""
    s = dt_text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    m = re.match(r"(.*?)([+-])(\d{2})(:?)(\d{2})$", s)  # +0200 → +02:00
    if m and m.group(4) == "":
        s = f"{m.group(1)}{m.group(2)}{m.group(3)}:{m.group(5)}"
    for pat in _EXIF_PATTERNS:
        try:
            return datetime.strptime(s, pat)
        except Exception:
            continue
    return None


def extract_exif_create_dates_batch(media_paths: List[Path]) -> Dict[Path, Optional[str]]:
    """
    Batch version of extract_exif_create_date: calls exiftool once per chunk and returns a mapping.
    It tries CreateDate, then DateTimeOriginal (with OffsetTimeOriginal if available), then ModifyDate.
    """
    results: Dict[Path, Optional[str]] = {p: None for p in media_paths}
    if not media_paths:
        return results
    # If exiftool is not available, bail out quickly
    if shutil.which(EXIFTOOL) is None:
        return results

    # exiftool -j produces JSON per file; we request the three tags we care about.
    args_base = [
        EXIFTOOL, "-j",
        "-CreateDate",
        "-DateTimeOriginal",
        "-OffsetTimeOriginal",
        "-ModifyDate",
    ]

    for batch in _chunked(media_paths, EXIF_BATCH_SIZE):
        try:
            completed = subprocess.run(
                args_base + [str(p) for p in batch],
                capture_output=True, text=True, check=False
            )
            if completed.returncode != 0 and not completed.stdout.strip():
                # If exiftool failed outright, skip this batch; individual files remain None
                continue

            data = json.loads(completed.stdout) if completed.stdout.strip() else []
            # Build a quick lookup by SourceFile
            by_src: Dict[str, Dict] = {}
            for entry in data:
                src = entry.get("SourceFile") or entry.get("SourceFilePath")
                if src:
                    by_src[str(src)] = entry

            for p in batch:
                entry = by_src.get(str(p))
                if not entry:
                    continue
                # Resolve fields in priority order
                dt_text = entry.get("CreateDate") or entry.get("DateTimeOriginal") or entry.get("ModifyDate")
                if not dt_text:
                    continue

                parsed = _parse_exif_datetime(dt_text)
                if not parsed:
                    continue

                # If DateTimeOriginal had an explicit offset tag, prefer it
                tzinfo = None
                if entry.get("DateTimeOriginal"):
                    off = entry.get("OffsetTimeOriginal")
                    if off and re.match(r"^[+-]\d{2}:\d{2}$", str(off)):
                        try:
                            sign = 1 if str(off)[0] == "+" else -1
                            hours = int(str(off)[1:3])
                            minutes = int(str(off)[4:6])
                            tzinfo = timezone(sign * timedelta(hours=hours, minutes=minutes))
                        except Exception:
                            tzinfo = None

                if tzinfo is not None:
                    aware = parsed.replace(tzinfo=tzinfo)
                    results[p] = aware.isoformat()
                else:
                    results[p] = parsed.isoformat()
        except Exception:
            # Leave batch items as None if anything goes wrong
            continue

    return results


def find_json_files(root_directory: Path) -> List[Path]:
    """
    Recursively find all JSON sidecar files under root_directory (ending in .json),
    skipping directory-level metadata.json and ignorable suffixes and pruning noisy dirs.
    """
    root_directory = Path(root_directory)
    json_file_paths: List[Path] = []
    for root, dirs, files in os.walk(root_directory):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]  # prune
        root_p = Path(root)
        for name in files:
            if name in SKIP_FILES or not name.lower().endswith(".json"):
                continue
            if name.lower() == "metadata.json":
                continue
            if any(name.lower().endswith(sfx) for sfx in IGNORE_JSON_SUFFIXES):
                continue
            json_file_paths.append(root_p / name)
    return json_file_paths


_CACHED_JSON_INDEX: Optional[Dict[str, List[Path]]] = None
_CACHED_JSON_SET: Optional[Set[str]] = None


def _normalize_title_to_stems(title: str) -> List[str]:
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


def build_json_title_index(json_file_paths: List[Path]) -> Dict[str, List[Path]]:
    if not json_file_paths:
        return {}
    def parse_one(jpath: Path) -> Optional[Tuple[List[str], Path]]:
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
            stems = _normalize_title_to_stems(data.get("title", ""))
            if stems:
                return stems, jpath
        except Exception:
            return None
        return None

    index: Dict[str, List[Path]] = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        for result in tqdm(ex.map(parse_one, json_file_paths), total=len(json_file_paths),
                           desc="🧩 Indexing JSON titles", unit="file"):
            if not result:
                continue
            stems, jpath = result
            for s in stems:
                index.setdefault(s, []).append(jpath)
    return index


def _ensure_json_index(json_file_paths: List[Path]) -> Dict[str, List[Path]]:
    global _CACHED_JSON_INDEX, _CACHED_JSON_SET
    current_set = {str(p) for p in json_file_paths}
    if _CACHED_JSON_INDEX is None or _CACHED_JSON_SET != current_set:
        _CACHED_JSON_INDEX = build_json_title_index(json_file_paths)
        _CACHED_JSON_SET = current_set
    return _CACHED_JSON_INDEX


def match_json_from_index(media_file_path: Path, title_index: Dict[str, List[Path]]) -> Optional[Path]:
    """
    Match a media file to its JSON sidecar using a prebuilt title index.
    """
    media_keys = _normalize_title_to_stems(media_file_path.name)
    if not media_keys:
        return None
    candidates: List[Path] = []
    for key in media_keys:
        paths = title_index.get(key)
        if paths:
            candidates.extend(paths)
    # Deduplicate while preserving order
    seen: Set[str] = set()
    ordered: List[Path] = []

    for p in candidates:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            ordered.append(p)
    for p in ordered:
        if p.exists():
            return p
    return ordered[0] if ordered else None


def match_json_for_media(
    media_file_path: Path,
    json_file_paths: List[Path],
    title_index: Optional[Dict[str, List[Path]]] = None,
) -> Optional[Path]:
    """
    Match a media file to its JSON sidecar.
    1) Try exact filename.json match (case-insensitive).
    2) Fall back to title-index match if provided or cached.
    """
    # 1) Direct filename.json (fast path)
    candidate_name = (media_file_path.name + ".json").lower()
    for p in json_file_paths:
        if p.name.lower() == candidate_name and p.exists():
            return p

    # 2) Title-based match
    if title_index is None:
        title_index = _ensure_json_index(json_file_paths)
    return match_json_from_index(media_file_path, title_index)


def extract_exif_create_date(media_file_path: Path) -> Optional[str]:
    """
    Use ExifTool to read EXIF date tags in a single call and return ISO format.
    Returns None if not found or on error.
    """
    if shutil.which(EXIFTOOL) is None:
        return None
    logger.debug(f"Extracting EXIF CreateDate for {media_file_path}")
    try:
        out = subprocess.check_output(
            [EXIFTOOL, "-s3",
             "-CreateDate", "-DateTimeOriginal", "-OffsetTimeOriginal", "-ModifyDate",
             str(media_file_path)],
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()
        create, dto, off, mod = (out + ["", "", "", ""])[:4]
        dt_text = create or dto or mod
        if not dt_text:
            logger.debug(f"No EXIF CreateDate/DateTimeOriginal/ModifyDate found for {media_file_path}")
            return None

        parsed = _parse_exif_datetime(dt_text)
        if not parsed:
            return None

        # If DateTimeOriginal provided an explicit offset, prefer it
        if dto and off and re.match(r"^[+-]\d{2}:\d{2}$", off):
            try:
                sign = 1 if off[0] == "+" else -1
                hours = int(off[1:3])
                minutes = int(off[4:6])
                tzinfo = timezone(sign * timedelta(hours=hours, minutes=minutes))
                parsed = parsed.replace(tzinfo=tzinfo)
            except Exception:
                pass

        return parsed.isoformat()
    except Exception:
        return None


def extract_json_taken_date(media_file_path: Path, json_file_paths: List[Path], target_root: Path, title_index: Optional[Dict[str, List[Path]]] = None) -> Optional[str]:
    """
    Read JSON sidecar's photoTakenTime or creationTime timestamp by matching
    via internal title map. Returns UTC ISO string or None.

    Sidecars are always moved under the provided `target_root`:
      - <target_root>/json_processed on success
      - <target_root>/json_error on failure
    """
    # Find the correct JSON sidecar
    matched_json = match_json_for_media(media_file_path, json_file_paths, title_index=title_index)
    if not matched_json:
        logger.debug(f"No JSON sidecar matched for {media_file_path}")
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
            logger.debug(f"No timestamp field in JSON sidecar {json_path}")
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
                logger.error(f"Failed to move JSON to processed: {json_path} -> {destination}")
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
                logger.error(f"Failed to move JSON to error: {json_path} -> {destination}")
        logger.info(f"Moved JSON to error: {destination}")
        return None


def parse_date_from_filename(filename_stem: str) -> Optional[str]:
    """
    Parse a fallback date from filename patterns like YYYYMMDD_HHMMSS.
    Returns an ISO8601 date string or None.
    """

    match = FILENAME_DATE_PATTERN.search(filename_stem)
    if not match:
        logger.debug(f"No date pattern found in filename: {filename_stem}")
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
        logger.error(f"Failed to parse date from filename: {filename_stem}")
        return None


def _unique_renamed_path(original: Path, preferred_ext: str) -> Path:
    """Return a collision-free path when changing `original`'s suffix to `preferred_ext`."""
    target = original.with_suffix(preferred_ext)
    try:
        if target.exists() and original.exists():
            try:
                if target.samefile(original):
                    return target
            except Exception:
                pass
    except Exception:
        pass
    if not target.exists():
        return target
    # Find a unique `(n)` suffix
    stem = target.stem
    parent = target.parent
    index = 1
    while True:
        candidate = parent / f"{stem} ({index}){preferred_ext}"
        if not candidate.exists():
            return candidate
        index += 1


# Detect and correct file extension based on MIME type
def correct_file_extension_by_mime(media_file_path: Path) -> Path:
    """
    Detect the file's MIME type via `file --mime-type` when available; otherwise fall back to
    mimetypes-based guessing. If the current suffix doesn't match the preferred extension,
    rename (or copy+remove on cross-device) and return the new Path.
    """
    preferred_ext = None
    try:
        # Prefer the `file` utility if available — more accurate than extension-based guessing
        if shutil.which(FILE_CMD) is not None:
            mime = subprocess.check_output(
                [FILE_CMD, "--mime-type", "-b", str(media_file_path)],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            preferred_ext = MIME_EXTENSION_MAP.get(mime)
        else:
            # Fallback to extension-based guess
            guessed = fast_guess_mime(str(media_file_path))
            if guessed:
                preferred_ext = MIME_EXTENSION_MAP.get(guessed)

        if preferred_ext and media_file_path.suffix.lower() != preferred_ext:
            new_path = _unique_renamed_path(media_file_path, preferred_ext)
            try:
                media_file_path.rename(new_path)
            except Exception:
                # Cross-device or permission issues: copy+remove as fallback
                try:
                    new_path.write_bytes(media_file_path.read_bytes())
                    media_file_path.unlink(missing_ok=True)
                except Exception:
                    logger.error(f"Failed to change extension for {media_file_path} -> {new_path}")
                    return media_file_path
            return new_path
    except Exception:
        # Non-fatal: leave file as-is if we couldn't determine MIME or rename failed
        pass
    return media_file_path
