#!/usr/bin/env python3
"""
dedupe.py

Detect and optionally move duplicate or near-duplicate images in a directory
using perceptual hashes (pHash & dHash) plus SSIM for visual similarity.

Usage:
  # This module expects an existing SQLite connection and does not open one itself.
"""

from pathlib import Path

from PIL import Image
from imagehash import phash, dhash
import cv2
from skimage.metrics import structural_similarity as ssim

import logging

# Ensure a logs directory exists next to this script
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_file = LOGS_DIR / "hash_duplicates.log"
handler = logging.FileHandler(log_file, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# Tunable thresholds
PHASH_THRESHOLD = 3     # hamming distance
DHASH_THRESHOLD = 3     # hamming distance
SSIM_THRESHOLD  = 0.95  # SSIM score (0–1)


# Perceptual hash utilities
def compute_perceptual_hashes(media_file_path: Path) -> tuple[str, str]:
    """
    Compute perceptual pHash and dHash for the given media file.
    Returns a tuple (pHash, dHash) as hex strings, or (None, None) on error.
    """
    try:
        image = Image.open(media_file_path).convert("RGB")
        return str(phash(image)), str(dhash(image))
    except Exception as err:
        logger.error(f"Failed computing perceptual hashes for {media_file_path}: {err}")
        return None, None


def compute_ssim(path_a: Path, path_b: Path) -> float:
    """Return SSIM score between two images, after resizing to match.
        SSIM ≥ 0.98 ⇒ near-identical pixels (an “edit”)
	    SSIM < 0.95 ⇒ probably a genuinely different shot    
    """
    a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return 0.0
    # resize both to the same smallest shape
    height, width = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a_small = cv2.resize(a, (width, height))
    b_small = cv2.resize(b, (width, height))
    score, _ = ssim(a_small, b_small, full=True)
    return score


def make_unique_path(parent_directory: Path, base_name: str, extension: str, original_file_path: Path = None) -> Path:
    """
    Given a base_name and extension, return a unique Path in parent_directory:
    base_name+extension or base_name(n)+extension if there’s a collision.
    """
    candidate = parent_directory / f"{base_name}{extension}"
    if not candidate.exists() or (original_file_path and candidate.samefile(original_file_path)):
        return candidate
    index = 1
    while True:
        numbered = parent_directory / f"{base_name}({index}){extension}"
        if not numbered.exists():
            return numbered
        index += 1


def find_duplicates(connection, root_dir: Path):
    """
    Scan root_dir for duplicate images using perceptual hashes and file sizes
    stored in the provided SQLite connection's 'media' table.
    """
    # 1) Query catalog for files with hashes
    cursor = connection.cursor()
    cursor.execute(
        "SELECT filepath, phash, dhash FROM media WHERE phash IS NOT NULL AND dhash IS NOT NULL"
    )
    entries = []
    for filepath_str, phash_str, dhash_str in cursor.fetchall():
        file_path = Path(filepath_str)
        if file_path.exists():
            entries.append((file_path, phash_str, dhash_str))

    duplicates = []
    visited = set()

    # 2) Compare each file to subsequent ones
    for i, (first, phash1_str, dhash1_str) in enumerate(entries):
        if first in visited:
            continue
        ph1 = int(phash1_str, 16)
        dh1 = int(dhash1_str, 16)
        for second, phash2_str, dhash2_str in entries[i+1:]:
            if second in visited:
                continue
            ph2 = int(phash2_str, 16)
            dh2 = int(dhash2_str, 16)

            # 2a) Check Hamming distances
            if bin(ph1 ^ ph2).count("1") <= PHASH_THRESHOLD and bin(dh1 ^ dh2).count("1") <= DHASH_THRESHOLD:
                # 2b) Verify with SSIM
                sim_score = compute_ssim(first, second)
                if sim_score >= SSIM_THRESHOLD:
                    duplicates.append((first, second, sim_score))
                    visited.add(second)

        visited.add(first)

    # 3) Rename
    for keep, dup, score in duplicates:
        # Generate a unique duplicate filename in the same folder
        new_path = make_unique_path(
            dup.parent,
            dup.stem + "_duplicate",
            dup.suffix,
            original_file_path=dup
        )
        try:
            dup.rename(new_path)
            logger.info(f"Marked duplicate {dup} → {new_path}")
            print(f"🔖 Marked duplicate: {new_path}  ← visually matches {keep}  (SSIM={score:.3f})")
        except Exception as error:
            logger.error(f"Failed renaming duplicate {dup}: {error}")
    print(f"\n⚠️  Found and renamed {len(duplicates)} duplicate(s).")

    return duplicates
