#!/usr/bin/env python3
"""
dedupe.py

Detect and optionally move duplicate or near-duplicate images in a directory
using perceptual hashes (pHash & dHash) plus SSIM for visual similarity.

Usage:
  # Print duplicates only
  python dedupe.py /path/to/organized

  # Move all duplicates (keeping the first-occurrence) into ./duplicates
  python dedupe.py /path/to/organized --move ./duplicates
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image
import numpy as np
from imagehash import phash, dhash
import cv2
from skimage.metrics import structural_similarity as ssim

# Tunable thresholds
PHASH_THRESHOLD = 3     # hamming distance
DHASH_THRESHOLD = 3     # hamming distance
SSIM_THRESHOLD  = 0.98  # SSIM score (0–1)


def compute_hashes(image_path: Path):
    """Return (pHash, dHash) for the given image."""
    img = Image.open(image_path).convert("RGB")
    return phash(img), dhash(img)


def compute_ssim(path_a: Path, path_b: Path) -> float:
    """Return SSIM score between two images, after resizing to match."""
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


def find_duplicates(root_dir: Path, move_to: Path = None):
    """
    Scan root_dir for images, compare each pair by pHash/dHash + SSIM,
    and either print duplicates or move them to move_to.
    """
    # 1) Gather all image files
    media_extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif"}
    files = [p for p in root_dir.rglob("*") if p.suffix.lower() in media_extensions]

    # 2) Compute perceptual hashes for each file
    hashes = {}
    for path in files:
        try:
            hashes[path] = compute_hashes(path)
        except Exception as err:
            print(f"⚠️  Failed hashing {path.name}: {err}")

    duplicates = []
    visited = set()

    # 3) Compare each file to subsequent ones
    for i, first in enumerate(files):
        if first in visited:
            continue
        ph1, dh1 = hashes.get(first, (None, None))
        for second in files[i+1:]:
            if second in visited:
                continue
            ph2, dh2 = hashes.get(second, (None, None))
            if ph1 is None or ph2 is None:
                continue

            # 3a) Check Hamming distances
            if abs(ph1 - ph2) <= PHASH_THRESHOLD and abs(dh1 - dh2) <= DHASH_THRESHOLD:
                # 3b) Verify with SSIM
                sim_score = compute_ssim(first, second)
                if sim_score >= SSIM_THRESHOLD:
                    duplicates.append((first, second, sim_score))
                    visited.add(second)

        visited.add(first)

    # 4) Report or move
    if move_to:
        move_to.mkdir(parents=True, exist_ok=True)
        for keep, dup, score in duplicates:
            dest = move_to / dup.name
            shutil.move(str(dup), str(dest))
        print(f"➡️  Moved {len(duplicates)} duplicates into {move_to}")
    else:
        for keep, dup, score in duplicates:
            print(f"Duplicate: {dup}  ← visually matches {keep}  (SSIM={score:.3f})")
        print(f"\n⚠️  Found {len(duplicates)} duplicate(s).")

    return duplicates


def main():
    parser = argparse.ArgumentParser(description="Find duplicate images by hash + SSIM")
    parser.add_argument("directory", help="Root directory of organized media")
    parser.add_argument(
        "--move", "-m",
        help="If set, move detected duplicates into this folder",
        dest="move_dir",
        metavar="DUP_DIR"
    )
    args = parser.parse_args()

    root = Path(args.directory).expanduser()
    if not root.is_dir():
        parser.error(f"Directory not found: {root}")

    move_target = Path(args.move_dir).expanduser() if args.move_dir else None
    find_duplicates(root, move_to=move_target)


if __name__ == "__main__":
    main()