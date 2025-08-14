#!/usr/bin/env python3
"""
Deduplicate and rename media files.

This module identifies visual duplicates using a perceptual-hash prefilter
(pHash + dHash → Hamming distance) and an SSIM verification step for images.
For videos, it relies on a perceptual **videohash** (computed earlier in the
pipeline) and flags near-duplicates by Hamming distance.

After classification, it renames **all** files to their `chosen_date` (already
formatted by `choose_timestamp.py`), appending "_duplicate" for files that
were marked as duplicates. Each rename is recorded in the database via the
`renamed_filepath` column.

Best‑practices applied:
- Strong type hints and cohesive helper functions
- Explicit constants for thresholds and paths
- Robust logging with a rotating file handler
- Clear separation of: loading entries → comparing → renaming
- Defensive updates to the catalog (SSIM stored for both image files)
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import sqlite3

import cv2
from PIL import Image
from imagehash import dhash, phash
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from catalog import update_renamed_filepath, update_ssim_score

# ----------------------------------
# Configuration & logging
# ----------------------------------
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "hash_duplicates.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # rotate at ~2MB, keep a couple of backups
    _fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)

# Tunable thresholds (images)
PHASH_THRESHOLD: int = 3      # Hamming distance threshold for pHash
DHASH_THRESHOLD: int = 3      # Hamming distance threshold for dHash
SSIM_THRESHOLD: float = 0.95  # SSIM threshold (0..1) for duplicates

# Tunable thresholds (videos)
VIDEO_HASH_THRESHOLD: int = 10  # Hamming distance threshold for videohash duplicates

# Bucket size for pHash prefix to reduce comparisons
PHASH_BUCKET_PREFIX: int = 4   # compare files only within same leading hex prefix


# ----------------------------------
# Data structures
# ----------------------------------
@dataclass(frozen=True)
class HashedEntry:
    path: Path
    phash_hex: str
    dhash_hex: str


@dataclass(frozen=True)
class DuplicateRecord:
    keep_path: Path
    duplicate_path: Path
    score: float  # SSIM (images) or 1.0 for video hash decision


# ----------------------------------
# Utilities
# ----------------------------------

def _hamming_hex(a_hex: str, b_hex: str) -> int:
    """Return Hamming distance between two same-length hex strings."""
    try:
        return (int(a_hex, 16) ^ int(b_hex, 16)).bit_count()
    except Exception:
        return 64  # large distance if parsing failed


def compute_perceptual_hashes(media_file_path: Path) -> Tuple[str | None, str | None]:
    """Compute pHash and dHash hex strings for an image path, or (None, None) on error."""
    try:
        image = Image.open(media_file_path).convert("RGB")
        return str(phash(image)), str(dhash(image))
    except Exception as error:  # pragma: no cover (best-effort logging)
        logger.error(f"Failed computing perceptual hashes for {media_file_path}: {error}")
        return None, None


def compute_ssim_score(path_a: Path, path_b: Path) -> float:
    """Compute SSIM between two images after resizing to a common size.

    Returns 0.0 if either image cannot be read. Uses data_range=255 for uint8.
    """
    a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0.0
    # Resize both to the smaller common dimensions to compare like with like
    height = min(a.shape[0], b.shape[0])
    width = min(a.shape[1], b.shape[1])
    if height <= 0 or width <= 0:
        return 0.0
    a_small = cv2.resize(a, (width, height), interpolation=cv2.INTER_AREA)
    b_small = cv2.resize(b, (width, height), interpolation=cv2.INTER_AREA)
    try:
        return float(ssim(a_small, b_small, data_range=255))
    except Exception:
        return 0.0


def make_unique_path(parent_directory: Path, base_name: str, extension: str, original_file_path: Path | None = None) -> Path:
    """Return a unique path `base_name+extension` in `parent_directory`.

    If a collision exists, append `(n)` before the extension. If `candidate` is
    the same file as `original_file_path`, return the candidate (no-op).
    """
    candidate = parent_directory / f"{base_name}{extension}"
    try:
        if not candidate.exists() or (original_file_path and candidate.exists() and candidate.samefile(original_file_path)):
            return candidate
    except Exception:
        if not candidate.exists():
            return candidate
    index = 1
    while True:
        numbered = parent_directory / f"{base_name}({index}){extension}"
        if not numbered.exists():
            return numbered
        index += 1


# ----------------------------------
# Core pipeline steps (images)
# ----------------------------------

def _load_hashed_entries(connection: sqlite3.Connection) -> List[HashedEntry]:
    """Load media rows that have both pHash and dHash from the catalog."""
    cursor = connection.cursor()
    cursor.execute("SELECT filepath, phash, dhash FROM media WHERE phash IS NOT NULL AND dhash IS NOT NULL")
    results: List[HashedEntry] = []
    for filepath_str, phash_hex, dhash_hex in cursor.fetchall():
        path = Path(filepath_str)
        if path.exists() and phash_hex and dhash_hex:
            results.append(HashedEntry(path=path, phash_hex=str(phash_hex), dhash_hex=str(dhash_hex)))
    return results


def _bucket_by_phash_prefix(entries: Sequence[HashedEntry], prefix_len: int = PHASH_BUCKET_PREFIX) -> Dict[str, List[HashedEntry]]:
    """Group entries by the first `prefix_len` hex characters of the pHash."""
    buckets: Dict[str, List[HashedEntry]] = {}
    for entry in entries:
        key = entry.phash_hex[:max(0, prefix_len)]
        buckets.setdefault(key, []).append(entry)
    return buckets


def _find_duplicate_pairs(connection: sqlite3.Connection, entries: Sequence[HashedEntry]) -> List[DuplicateRecord]:
    """Return a list of DuplicateRecord identified among `entries` (images only).

    Strategy: Within each pHash prefix bucket, compare pairs with small
    Hamming distance on both pHash and dHash, then verify by SSIM. Record SSIM
    for both files, regardless of duplicate decision, for later analysis.
    """
    duplicates: List[DuplicateRecord] = []
    visited: Set[Path] = set()

    buckets = _bucket_by_phash_prefix(entries)

    for _key, bucket in tqdm(buckets.items(), desc="🔍 Comparing images", unit="bucket"):
        # local nested loop within bucket
        for i, left in enumerate(bucket):
            if left.path in visited:
                continue
            for right in bucket[i + 1 : ]:
                if right.path in visited:
                    continue

                # Quick gates: Hamming distance thresholds
                if _hamming_hex(left.phash_hex, right.phash_hex) > PHASH_THRESHOLD:
                    continue
                if _hamming_hex(left.dhash_hex, right.dhash_hex) > DHASH_THRESHOLD:
                    continue

                # Verify: SSIM (store it for both rows)
                ssim_score = compute_ssim_score(left.path, right.path)
                logger.debug(f"SSIM {ssim_score:.3f}: {left.path} vs {right.path}")
                try:
                    _ = update_ssim_score(connection, left.path, ssim_score)
                    _ = update_ssim_score(connection, right.path, ssim_score)
                except Exception as error:  # pragma: no cover
                    logger.error(f"SSIM DB update failed for {left.path} / {right.path}: {error}")

                if ssim_score >= SSIM_THRESHOLD:
                    duplicates.append(DuplicateRecord(keep_path=left.path, duplicate_path=right.path, score=ssim_score))
                    visited.add(right.path)
            visited.add(left.path)

    return duplicates


# ----------------------------------
# Core pipeline steps (videos)
# ----------------------------------

def _load_video_hash_entries(connection: sqlite3.Connection) -> List[Tuple[Path, str]]:
    """Load rows that have a video_hash set and still exist on disk."""
    cursor = connection.cursor()
    cursor.execute("SELECT filepath, video_hash FROM media WHERE video_hash IS NOT NULL")
    items: List[Tuple[Path, str]] = []
    for fp_str, vhash in cursor.fetchall():
        p = Path(fp_str)
        if p.exists() and vhash:
            items.append((p, str(vhash)))
    return items


def _find_duplicate_videos(connection: sqlite3.Connection) -> List[DuplicateRecord]:
    """Identify duplicate videos using videohash Hamming distance only."""
    pairs = _load_video_hash_entries(connection)
    if not pairs:
        return []

    # bucket videos by prefix to reduce comparisons
    buckets: Dict[str, List[Tuple[Path, str]]] = {}
    for p, h in pairs:
        buckets.setdefault(h[:4], []).append((p, h))

    duplicates: List[DuplicateRecord] = []
    visited: Set[Path] = set()
    for _key, bucket in tqdm(buckets.items(), desc="🎞️ Comparing videos", unit="bucket"):
        for i, (a_path, a_hash) in enumerate(bucket):
            if a_path in visited:
                continue
            for b_path, b_hash in bucket[i+1:]:
                if b_path in visited:
                    continue
                if _hamming_hex(a_hash, b_hash) <= VIDEO_HASH_THRESHOLD:
                    duplicates.append(DuplicateRecord(keep_path=a_path, duplicate_path=b_path, score=1.0))
                    visited.add(b_path)
            visited.add(a_path)

    return duplicates


# ----------------------------------
# Renaming
# ----------------------------------

def _rename_all_by_chosen_date(
    connection: sqlite3.Connection,
    duplicates: Sequence[DuplicateRecord],
) -> int:
    """Rename all files to `chosen_date` as base name; mark duplicates with `_duplicate`.

    Returns the number of files actually renamed.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT filepath, chosen_date FROM media")
    rows: List[Tuple[str, str | None]] = cursor.fetchall()

    duplicate_set: Set[Path] = {rec.duplicate_path for rec in duplicates}

    renamed_count = 0
    for filepath_str, chosen_str in tqdm(rows, desc="✏️ Renaming by chosen_date", unit="file"):
        file_path = Path(filepath_str)
        if not file_path.exists():
            logger.warning(f"File not found on disk, skipping rename: {file_path}")
            continue
        if not chosen_str:
            logger.warning(f"No chosen_date in DB for {file_path}; skipping rename")
            continue

        base_name = chosen_str
        if file_path in duplicate_set:
            base_name = f"{base_name}_duplicate"

        extension = file_path.suffix.lower()
        new_path = make_unique_path(file_path.parent, base_name, extension, original_file_path=file_path)

        if new_path != file_path:
            try:
                file_path.rename(new_path)
                # Keep catalog in sync
                ok = update_renamed_filepath(connection, file_path, new_path)
                if not ok:
                    logger.warning(f"DB row not matched when updating renamed_filepath for: {file_path}")
                logger.info(f"Renamed {file_path} → {new_path}")
                renamed_count += 1
            except Exception as error:  # pragma: no cover
                logger.error(f"Failed to rename {file_path}: {error}")

    return renamed_count


# ----------------------------------
# Public API
# ----------------------------------

def find_duplicates(connection: sqlite3.Connection, root_dir: Path) -> List[DuplicateRecord]:
    """End‑to‑end dedupe: gather hashed entries → find matches → rename everything.

    `root_dir` is accepted for API compatibility but not used directly because
    we rely on the catalog’s file list. The function returns the list of
    `DuplicateRecord` found.
    """
    # Image duplicates
    image_entries = _load_hashed_entries(connection)
    if not image_entries:
        logger.info("No hashed image entries present; continuing with videos only.")
    image_duplicates = _find_duplicate_pairs(connection, image_entries) if image_entries else []

    # Video duplicates (based on videohash)
    video_duplicates = _find_duplicate_videos(connection)

    # Merge both
    all_duplicates = [*image_duplicates, *video_duplicates]

    renamed = _rename_all_by_chosen_date(connection, all_duplicates)
    print(f"\n🔖 Renamed {renamed} file(s) (duplicates marked with _duplicate).")

    return all_duplicates
