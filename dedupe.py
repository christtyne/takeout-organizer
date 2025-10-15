#!/usr/bin/env python3
"""
Deduplicate and rename media files.

This module identifies visual duplicates using a perceptual-hash prefilter
(pHash + dHash → Hamming distance) and an SSIM verification step for images.
For videos, it relies on a perceptual **videohash** (computed earlier in the
pipeline) and flags near-duplicates by Hamming distance.

After classification, it renames **all** files to their `chosen_timestamp` (already
formatted by `choose_timestamp.py`), appending "_duplicate" for files that
were marked as duplicates. Each rename is recorded in the database via the
`renamed_filepath` column.

Best‑practices applied:
- Strong type hints and cohesive helper functions
- Explicit constants for thresholds and paths
- Centralized per-module file logger (no duplicate handlers)
- Clear separation of: loading entries → comparing → renaming
- Defensive updates to the catalog (SSIM stored for both image files)
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import os
import glob
import shutil
import subprocess
import tempfile

import cv2
from PIL import Image, ImageOps

# --- Pillow 10 compatibility shim ---
# Some libraries (e.g., videohash<=X) still reference Image.ANTIALIAS, which
# was removed in Pillow 10 in favor of Image.Resampling.LANCZOS.
try:
    from PIL.Image import Resampling  # Pillow >= 9.1 provides this
    if not hasattr(Image, "ANTIALIAS"):
        # Back-compat alias for libs expecting ANTIALIAS
        Image.ANTIALIAS = Resampling.LANCZOS  # type: ignore[attr-defined]
except Exception:
    # On older Pillow versions, Image.ANTIALIAS already exists; ignore.
    pass

from imagehash import dhash, phash, whash
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from catalog import ensure_indexes, batch_update_ssim, batch_update_renamed_filepath
from videohash import VideoHash


# ----------------------------------
# Configuration & logging
# ----------------------------------
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"{Path(__file__).stem}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)  # only log errors and above
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(LOG_FILE) for h in logger.handlers):
    _fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.ERROR)  # ensure handler filters to errors
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_fh)


# Tunable thresholds (images)
PHASH_THRESHOLD: int = 3      # Hamming distance threshold for pHash
DHASH_THRESHOLD: int = 3      # Hamming distance threshold for dHash
SSIM_THRESHOLD: float = 0.95  # SSIM threshold (0..1) for duplicates

# Tunable thresholds (videos)
VIDEO_HASH_THRESHOLD: int = 10  # Hamming distance threshold for videohash duplicates

# A softer near-miss window that still triggers SSIM verification
NEAR_MISS_THRESHOLD: int = 8    # if any hash distance <= 8, we still verify via SSIM
# Wavelet-hash tolerance helps when compression/contrast changes make pHash/dHash diverge
WHASH_THRESHOLD: int = 6

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
    """Compute pHash and dHash hex strings for an image path, or (None, None) on error.

    Uses EXIF-based auto-orientation to avoid false negatives.
    """
    try:
        image = Image.open(media_file_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return str(phash(image)), str(dhash(image))
    except Exception as error:  # pragma: no cover (best-effort logging)
        logger.error(f"Failed computing perceptual hashes for {media_file_path}: {error}")
        return None, None



def _compute_whash_hex(media_file_path: Path) -> str | None:
    try:
        image = Image.open(media_file_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return str(whash(image))
    except Exception:
        return None


# --- Cached helpers for geometry and wavelet hash ---
@lru_cache(maxsize=100_000)
def _compute_whash_hex_cached(path_str: str) -> str | None:
    """Memoized wHash to avoid recomputation during near-miss checks."""
    try:
        return _compute_whash_hex(Path(path_str))
    except Exception:
        return None

@lru_cache(maxsize=100_000)
def get_image_shape(path_str: str) -> Tuple[int, int] | None:
    """
    Fast header-only image dimension lookup using Pillow.
    Returns (height, width) or None on failure.
    """
    try:
        with Image.open(path_str) as im:
            w, h = im.size
            return (h, w)
    except Exception:
        return None



# ----------------------------------
# Video hashing helpers
# ----------------------------------

def _combine_frame_hashes(hex_hashes: List[str]) -> str:
    """Combine multiple 64‑bit hex hashes (e.g., pHashes) via majority vote per bit."""
    ints = [int(h, 16) for h in hex_hashes if h]
    if not ints:
        return ""
    bit_counts = [0] * 64
    for value in ints:
        for bit in range(64):
            if (value >> bit) & 1:
                bit_counts[bit] += 1
    threshold = len(ints) / 2.0
    result = 0
    for bit in range(64):
        if bit_counts[bit] > threshold:
            result |= (1 << bit)
    return f"{result:016x}"


def compute_video_hash(media_file_path: Path) -> str | None:
    """Compute a perceptual video hash using videohash; fallback to manual frame hashing.

    Primary: videohash.VideoHash (may fail for odd crops on tiny videos).
    Fallback: extract up to 8 frames with ffmpeg (no crop), pHash each frame,
    then majority‑vote the 64‑bit bits into one hex digest.
    """
    # --- Primary path: videohash ---
    try:
        vh = VideoHash(path=str(media_file_path))
        value = getattr(vh, "hash", None) or getattr(vh, "hash_hex", None)
        if value:
            return str(value)
    except Exception as error:  # pragma: no cover
        logger.error(f"Failed computing video hash for {media_file_path}: {error}")
        # Continue to fallback

    # --- Fallback path: ffmpeg frame sampling + image pHash combine ---
    if shutil.which("ffmpeg") is None:
        logger.error("Fallback video hashing unavailable: ffmpeg not found in PATH")
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="vh_fallback_") as tmpdir:
            frame_pattern = os.path.join(tmpdir, "frame_%03d.jpg")
            # safe filters: 1 fps sampling, square scale, no crop
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(media_file_path),
                "-vf", "fps=1,scale=160:160:flags=bicubic",
                "-frames:v", "8",
                frame_pattern,
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", "ignore")
                logger.error(f"Fallback ffmpeg extraction failed for {media_file_path}: {stderr}")
                return None

            frames = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
            if not frames:
                logger.error(f"No frames extracted in fallback for {media_file_path}")
                return None

            hex_hashes: List[str] = []
            for frame_path in frames:
                try:
                    img = Image.open(frame_path).convert("RGB")
                    hex_hashes.append(str(phash(img)))
                except Exception as e:  # pragma: no cover
                    logger.debug(f"Skipping frame {frame_path}: {e}")

            combined = _combine_frame_hashes(hex_hashes)
            if not combined:
                logger.error(f"Failed to combine frame hashes for {media_file_path}")
                return None

            logger.info(f"Used fallback video hash for {media_file_path}")
            return combined
    except Exception as error:  # pragma: no cover
        logger.error(f"Video hashing fallback crashed for {media_file_path}: {error}")
        return None


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
    

def make_unique_path(dest_dir: Path, stem: str, suffix: str, original: Path | None = None) -> Path:
    candidate = dest_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    # allow if it’s the same file (e.g., case-only change)
    try:
        if original and candidate.exists() and candidate.samefile(original):
            return candidate
    except Exception:
        pass
    i = 1
    while True:
        numbered = dest_dir / f"{stem} ({i}){suffix}"
        if not numbered.exists():
            return numbered
        i += 1


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
    # Sort buckets by size descending so we handle large ones first
    return dict(sorted(buckets.items(), key=lambda kv: len(kv[1]), reverse=True))


def _find_duplicate_pairs(connection: sqlite3.Connection, entries: Sequence[HashedEntry]) -> List[DuplicateRecord]:
    """Return DuplicateRecord list identified among `entries` (images only), parallelizing SSIM checks.

    Strategy
    --------
    1) Partition by pHash prefix to keep candidate sets small.
    2) Within each bucket, compare hashes to form a list of *candidates* that need SSIM.
       We only enqueue SSIM when (strict) both pHash and dHash within threshold OR
       (near_miss) min distance within NEAR_MISS_THRESHOLD **and** wHash also close.
    3) Compute SSIM in parallel using a ProcessPoolExecutor (CPU bound).
    4) Update the DB and assemble duplicates on the main process to keep the SQLite
       connection thread/fork safe.
    """
    duplicates: List[DuplicateRecord] = []
    visited: Set[Path] = set()

    buckets = _bucket_by_phash_prefix(entries)
    ssim_jobs: List[Tuple[Path, Path]] = []

    # --- Stage 1: hash-only prefilter to build SSIM jobs ---
    for _key, bucket in tqdm(buckets.items(), desc="🔍 Hash prefilter (images)", unit="bucket", leave=False):
        # Adaptively split very large buckets by a longer prefix to reduce O(n^2)
        buckets_to_process: List[List[HashedEntry]] = []
        if len(bucket) > 2000:
            sub: Dict[str, List[HashedEntry]] = {}
            for e in bucket:
                sub.setdefault(e.phash_hex[:6], []).append(e)
            buckets_to_process = list(sub.values())
        else:
            buckets_to_process = [bucket]

        for sub_bucket in buckets_to_process:
            local_visited: Set[Path] = set()
            for i, left in enumerate(sub_bucket):
                if left.path in visited or left.path in local_visited:
                    continue
                for right in sub_bucket[i + 1:]:
                    if right.path in visited or right.path in local_visited:
                        continue

                    d_ph = _hamming_hex(left.phash_hex, right.phash_hex)
                    d_dh = _hamming_hex(left.dhash_hex, right.dhash_hex)

                    strict_pass = (d_ph <= PHASH_THRESHOLD and d_dh <= DHASH_THRESHOLD)
                    if not strict_pass:
                        # Near-miss path: require a supportive wavelet-hash agreement
                        if min(d_ph, d_dh) > NEAR_MISS_THRESHOLD:
                            continue
                        w1 = _compute_whash_hex_cached(str(left.path))
                        w2 = _compute_whash_hex_cached(str(right.path))
                        if not (w1 and w2) or _hamming_hex(w1, w2) > WHASH_THRESHOLD:
                            continue

                    # Lightweight geometry check using cached header dimensions
                    try:
                        sh_left = get_image_shape(str(left.path))
                        sh_right = get_image_shape(str(right.path))
                        if not sh_left or not sh_right:
                            continue
                        ha, wa = sh_left
                        hb, wb = sh_right
                        if (max(ha, hb) / max(1, min(ha, hb)) > 3.0) or (max(wa, wb) / max(1, min(wa, wb)) > 3.0):
                            continue
                    except Exception:
                        pass

                    ssim_jobs.append((left.path, right.path))
                local_visited.add(left.path)

    # --- Stage 2: Parallel SSIM for candidates ---
    def _ssim_worker(args: Tuple[Path, Path]) -> Tuple[Path, Path, float]:
        a, b = args
        return a, b, compute_ssim_score(a, b)

    # Nothing to do?
    if not ssim_jobs:
        return []

    results: List[Tuple[Path, Path, float]] = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = {pool.submit(_ssim_worker, job): job for job in ssim_jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="🧮 SSIM (parallel)", unit="pair", leave=False):
            try:
                results.append(fut.result())
            except Exception:
                # Ignore failed jobs; treat as non-duplicates
                pass

    # --- Stage 3: DB updates and final decisions (main process) ---
    seen_pairs: Set[frozenset[Path]] = set()
    duplicates: List[DuplicateRecord] = []
    rows_to_update: Set[Tuple[float, str]] = set()

    for a_path, b_path, score in results:
        pair_key = frozenset((a_path, b_path))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        # accumulate SSIM updates for both files
        rows_to_update.add((score, str(a_path)))
        rows_to_update.add((score, str(b_path)))

        if score >= SSIM_THRESHOLD:
            duplicates.append(DuplicateRecord(keep_path=a_path, duplicate_path=b_path, score=score))
            visited.add(b_path)

    # Batch-commit SSIM scores (best-effort)
    if rows_to_update:
        try:
            batch_update_ssim(connection, list(rows_to_update))
        except Exception as error:  # pragma: no cover
            logger.error(f"SSIM DB batch update failed for {len(rows_to_update)} rows: {error}")

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

    # Mark exact hash duplicates first
    by_hash: Dict[str, List[Tuple[Path, str]]] = {}
    for p, h in pairs:
        by_hash.setdefault(h, []).append((p, h))

    duplicates: List[DuplicateRecord] = []
    visited: Set[Path] = set()
    for h, items in by_hash.items():
        if len(items) > 1:
            keep = items[0][0]
            for p, _ in items[1:]:
                duplicates.append(DuplicateRecord(keep_path=keep, duplicate_path=p, score=1.0))
                visited.add(p)

    # Continue with near-miss matching on the remaining items
    pairs = [(p, h) for p, h in pairs if p not in visited]
    if not pairs:
        return duplicates

    # bucket videos by prefix to reduce comparisons
    buckets: Dict[str, List[Tuple[Path, str]]] = {}
    for p, h in pairs:
        buckets.setdefault(h[:4], []).append((p, h))

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


def _find_duplicates_same_chosen_timestamp(connection: sqlite3.Connection) -> List[DuplicateRecord]:
    """Within each chosen_timestamp group, verify pairs by SSIM in parallel.

    This catches cases where hashing differs due to recompression/processing,
    but the timestamp base name collides.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT filepath, chosen_timestamp FROM media WHERE chosen_timestamp IS NOT NULL")
    rows = cursor.fetchall()

    # Build groups by chosen_timestamp
    groups: Dict[str, List[Path]] = {}
    for fp_str, chosen in rows:
        p = Path(fp_str)
        if p.exists() and chosen:
            groups.setdefault(chosen, []).append(p)

    # Build candidate pairs
    jobs: List[Tuple[Path, Path]] = []
    for paths in groups.values():
        if len(paths) < 2:
            continue
        # Heuristic: sort by file size (if available) so near sizes are adjacent and more likely
        try:
            paths = sorted(paths, key=lambda p: p.stat().st_size)
        except Exception:
            paths = list(paths)
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                a, b = paths[i], paths[j]
                jobs.append((a, b))

    if not jobs:
        return []

    def _ssim_worker(args: Tuple[Path, Path]) -> Tuple[Path, Path, float]:
        a, b = args
        return a, b, compute_ssim_score(a, b)

    results: List[Tuple[Path, Path, float]] = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        futures = {pool.submit(_ssim_worker, job): job for job in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="📆 SSIM same‑timestamp", unit="pair", leave=False):
            try:
                results.append(fut.result())
            except Exception:
                pass

    duplicates: List[DuplicateRecord] = []
    seen_pairs: Set[frozenset[Path]] = set()
    rows_to_update: Set[Tuple[float, str]] = set()

    for a, b, score in results:
        key = frozenset((a, b))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        rows_to_update.add((score, str(a)))
        rows_to_update.add((score, str(b)))

        if score >= SSIM_THRESHOLD:
            duplicates.append(DuplicateRecord(keep_path=a, duplicate_path=b, score=score))

    if rows_to_update:
        try:
            batch_update_ssim(connection, list(rows_to_update))
        except Exception as error:
            logger.error(f"SSIM DB batch update (same-timestamp) failed for {len(rows_to_update)} rows: {error}")

    return duplicates


# ----------------------------------
# Renaming
# ----------------------------------

def _rename_all_by_chosen_timestamp(
    connection: sqlite3.Connection,
    duplicates: Sequence[DuplicateRecord],
) -> int:
    """Rename all files to `chosen_timestamp` as base name; mark duplicates with `_duplicate`.

    Returns the number of files actually renamed.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT filepath, chosen_timestamp FROM media")
    rows: List[Tuple[str, str | None]] = cursor.fetchall()

    duplicate_set: Set[Path] = {rec.duplicate_path for rec in duplicates}

    renamed_count = 0
    for filepath_str, chosen_str in tqdm(rows, desc="✏️ Renaming by chosen_timestamp", unit="file"):
        file_path = Path(filepath_str)
        if not file_path.exists():
            logger.error(f"File not found on disk, skipping rename: {file_path}")
            continue
        if not chosen_str:
            logger.error(f"No chosen_timestamp in DB for {file_path}; skipping rename")
            continue

        base_name = chosen_str
        if file_path in duplicate_set:
            base_name = f"{base_name}_duplicate"

        extension = file_path.suffix.lower()
        # FIX: correct keyword argument is `original`, not `original_file_path`
        new_path = make_unique_path(file_path.parent, base_name, extension, original=file_path)

        if new_path != file_path:
            try:
                file_path.rename(new_path)
                # Use batch helper even for a single row for consistency
                try:
                    updated = batch_update_renamed_filepath(connection, [(str(new_path), str(file_path))])
                except Exception as e:
                    updated = 0
                    logger.error(f"DB update failed for renamed_filepath {file_path} → {new_path}: {e}")
                if updated == 0:
                    logger.error(f"DB row not matched when updating renamed_filepath for: {file_path}")
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
    This function now parallelizes SSIM across CPU cores and ensures helpful DB indexes exist.
    """
    ensure_indexes(connection)
    # Image duplicates
    image_entries = _load_hashed_entries(connection)
    if not image_entries:
        logger.info("No hashed image entries present; continuing with videos only.")
    image_duplicates = _find_duplicate_pairs(connection, image_entries) if image_entries else []

    # Video duplicates (based on videohash)
    video_duplicates = _find_duplicate_videos(connection)

    # Duplicates from identical chosen_timestamp groups (SSIM‑verified)
    date_group_duplicates = _find_duplicates_same_chosen_timestamp(connection)

    # Merge all sources
    all_duplicates = [*image_duplicates, *video_duplicates, *date_group_duplicates]

    renamed = _rename_all_by_chosen_timestamp(connection, all_duplicates)
    return all_duplicates
