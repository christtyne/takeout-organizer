#!/usr/bin/env python3
"""
takeout_setup.py

Orchestrator for the Google Takeout Photo Organizer pipeline.

Steps:
1. Initialize SQLite catalog.
2. Scan the target directory for media files and JSON sidecars.
3. Extract EXIF CreateDate and JSON photoTakenTime for each file.
4. Parse fallback dates from filenames.
5. Choose the final timestamp per file.
6. Rename or move files into date-based folders.
7. Optionally clean up empty directories.
"""

# Tunable batch sizes
EXIF_CHUNK = 1000
HASH_CHUNK = 1000

import os
import sys
import shutil
import subprocess
from pathlib import Path
import logging
import argparse

# Import project modules
import catalog
import extract_meta
import choose_timestamp
import reorganize
from clean_empty_folders import clean_empty_folders
import dedupe
import extract_archives

from tqdm import tqdm


# Local helper for chunked processing
def _chunked(seq, size):
    batch = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

# CLI arg parser for --resume and future flags
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google Takeout Photo Organizer")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages with no pending work (JSON association, timestamps, hashing).")
    # keep room for future flags like --dry-run, --skip-dedupe, etc.
    return parser.parse_args()


# Centralized logs directory at project root
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / f"{Path(__file__).stem}.log"

# Configure per-module file logger (idempotent)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(LOG_FILE) for h in logger.handlers):
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

if shutil.which("ffmpeg") is None:
    print("⚠️  ffmpeg not found — video hashing will be skipped.")

if shutil.which("exiftool") is None:
    print("⚠️  exiftool not found — EXIF batch extraction will be skipped where required.")

def prompt_for_directory(prompt: str) -> Path:
    """Open a native macOS 'choose folder' dialog and return a validated Path.

    Falls back to manual input only if AppleScript fails.
    """
    try:
        selected = subprocess.check_output([
            "osascript", "-e",
            f'POSIX path of (choose folder with prompt "{prompt}")'
        ], text=True).strip()
    except subprocess.CalledProcessError:
        selected = input(f"{prompt} (enter full path): ").strip()

    directory = Path(selected).expanduser()
    if directory.is_dir():
        return directory
    print(f"❌ Selected path is not a directory: {directory}", file=sys.stderr)
    logger.error(f"Invalid directory selected: {directory}")
    sys.exit(2)

def main():
    args = _parse_args()
    # 1) Decide extraction first, then pick folders
    if input("\n📦 Do you want to extract .tar/.tgz files? [y/N] ").strip().lower().startswith("y"):
        extract_directory = prompt_for_directory("📦 Select folder containing your .tar/.tgz files")
        target_directory = prompt_for_directory("📂 Select folder to extract your Google Takeout files")
        try:
            count = extract_archives.extract_all(extract_directory, target_directory)
            print(f"✅ Extracted {count} file(s)\n")
            logger.info(f"Extracted {count} file(s) from archives into {target_directory}")
        except Exception as error:
            print(f"❌ Extraction failed: {error}")
            logger.error(f"Extraction failed: {error}")
            sys.exit(2)
    else:
        # Use CLI arg/env if provided; otherwise ask for the existing Takeout folder
        if len(sys.argv) > 1:
            target_directory = Path(sys.argv[1]).expanduser()
        else:
            env_dir = os.getenv("TARGET_DIR", "")
            target_directory = Path(env_dir).expanduser() if env_dir else prompt_for_directory("📂 Select your Google Takeout folder")

        if not target_directory.is_dir():
            print(f"❌ Target directory is not a folder: {target_directory}")
            logger.error(f"Invalid target directory: {target_directory}")
            sys.exit(2)

    logger.info(f"Target directory: {target_directory}")

    # 2) Choose output directory
    output_directory = prompt_for_directory("💾 Select where organized files will be saved")
    logger.info(f"Output directory: {output_directory}")

    # 3) Initialize database
    print("\n🗃️  Creating database")
    connection = catalog.initialize_database()

    # Quick catalog health summary (non-fatal)
    try:
        print(catalog.health_check(connection, auto_fix=False))
    except Exception:
        logger.debug("Health check skipped (non-fatal).")

    # 4) Discover media files with progress bar
    print("\n🔎 Scanning for media files...")
    media_iterator = extract_meta.find_media_files(target_directory)
    media_file_paths = list(tqdm(media_iterator, desc="🖼️ Processing media files", unit="file"))
    print(f"✅ Found {len(media_file_paths)} media files")

    # 4b) Discover JSON sidecars with progress bar
    print("\n🧩 Scanning for JSON sidecars...")
    json_iterator = extract_meta.find_json_files(target_directory)
    json_file_paths = list(tqdm(json_iterator, desc="🧩 Processing JSON files", unit="file"))
    print(f"✅ Found {len(json_file_paths)} JSON files\n")
    no_json_available = len(json_file_paths) == 0

    logger.info(f"Found {len(media_file_paths)} media files and {len(json_file_paths)} JSON files")

    # Correct any mismatched extensions in place
    corrected_media_paths = []
    for media_file_path in tqdm(media_file_paths, desc="🛠️ Correcting extensions", unit="file"):
        corrected = extract_meta.correct_file_extension_by_mime(media_file_path)
        corrected_media_paths.append(corrected)
    media_file_paths = corrected_media_paths

    # 5) Populate catalog entries (single transaction for speed)
    connection.execute("BEGIN")
    try:
        for media in tqdm(media_file_paths, desc="🗃️ Populating catalog", unit="file"):
            catalog.add_media_entry(connection, media)  # idempotent
        connection.commit()
    except Exception:
        connection.rollback()
        raise

    # 6) Build JSON title index once, then associate sidecars fast (only missing)
    if no_json_available:
        print("⏭️  JSON association: no JSON sidecars found; skipping.")
    else:
        title_index = extract_meta.build_json_title_index(json_file_paths)
        pending_json = [Path(p) for p in catalog.fetch_without_json(connection)]
        if args.resume and not pending_json:
            print("⏭️  JSON association: nothing pending; skipping (resume).")
        else:
            connection.execute("BEGIN")
            try:
                for media_path in tqdm(pending_json, desc="🔗 Associating JSON (missing only)", unit="file"):
                    match = extract_meta.match_json_from_index(media_path, title_index)
                    if match:
                        catalog.update_json_path(connection, media_path, match)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    print(f"\n")

    # 7) Extract timestamps from EXIF and JSON (batched transactions; only where chosen timestamp is missing)
    pending_for_ts = [Path(p) for p in catalog.fetch_missing_timestamps(connection)]
    if args.resume and not pending_for_ts:
        print("⏭️  Timestamps: nothing pending; skipping (resume).")
    else:
        for batch in _chunked(pending_for_ts, EXIF_CHUNK):
            connection.execute("BEGIN")
            try:
                # Batch EXIF for speed, then per-file JSON taken date
                exif_map = extract_meta.extract_exif_create_dates_batch(batch)
                for media_file_path in tqdm(batch, desc="⏱️ Extracting timestamps (batch)", unit="file", leave=False):
                    create_date = exif_map.get(media_file_path)
                    if create_date:
                        catalog.update_exif_create_date(connection, media_file_path, create_date)
                    json_taken_date = extract_meta.extract_json_taken_date(media_file_path, json_file_paths, target_directory)
                    if json_taken_date:
                        catalog.update_json_taken_date(connection, media_file_path, json_taken_date)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    # 8) Fallback: parse date from filename (single transaction; only where chosen timestamp is missing)
    if args.resume and not pending_for_ts:
        print("⏭️  Filename parsing: nothing pending; skipping (resume).")
    else:
        connection.execute("BEGIN")
        try:
            for media_file_path in tqdm(pending_for_ts, desc="🔤 Parsing filenames (missing only)", unit="file"):
                filename_date = extract_meta.parse_date_from_filename(media_file_path.stem)
                if filename_date:
                    catalog.update_filename_parsed_date(connection, media_file_path, filename_date)
            connection.commit()
        except Exception:
            connection.rollback()
            raise


    # 9) Choose the final timestamp for each entry
    choose_timestamp.choose_timestamp_for_all(connection)

    # 10) Compute and store perceptual hashes (only missing, batched)

    # Pull pending items directly from the DB
    unhashed_images = [Path(p) for p in catalog.fetch_unhashed_images(connection)]
    unhashed_videos = [Path(p) for p in catalog.fetch_unhashed_videos(connection)]

    if args.resume and not unhashed_images and not unhashed_videos:
        print("⏭️  Hashing: nothing pending; skipping (resume).")
    else:

        # Images
        if unhashed_images:
            for batch in _chunked(unhashed_images, HASH_CHUNK):
                connection.execute("BEGIN")
                try:
                    for media_file_path in tqdm(batch, desc="🧮 Hashing images (batch)", unit="file", leave=False):
                        if not media_file_path.exists():
                            continue
                        perceptual_hash, difference_hash = dedupe.compute_perceptual_hashes(media_file_path)
                        if perceptual_hash:
                            catalog.update_phash(connection, media_file_path, perceptual_hash)
                        if difference_hash:
                            catalog.update_dhash(connection, media_file_path, difference_hash)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

        # Videos
        if unhashed_videos:
            for batch in _chunked(unhashed_videos, HASH_CHUNK):
                connection.execute("BEGIN")
                try:
                    for media_file_path in tqdm(batch, desc="🧮 Hashing videos (batch)", unit="file", leave=False):
                        if not media_file_path.exists():
                            continue
                        video_hash = dedupe.compute_video_hash(media_file_path)
                        if video_hash:
                            catalog.update_video_hash(connection, media_file_path, video_hash)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    # 11) Deduplication pass (images + videos)
    print("\n🔍 Running dedupe pass — images and videos")
    duplicates = dedupe.find_duplicates(connection, target_directory)
    print(f"🔎 Dedupe found {len(duplicates)} potential duplicate(s)")

    print(f"\n")


    # 12) Move files based on the chosen timestamp
    reorganize.reorganize_files(connection, output_directory)

    # 13) Optional cleanup of empty folders
    if input("\n🧹 Remove empty folders? [y/N] ").strip().lower().startswith("y"):
        clean_empty_folders(target_directory)

    # Database maintenance at the end
    catalog.vacuum_and_checkpoint(connection)

    # Final summary (non-fatal if any query fails)
    try:
        total_json_missing = connection.execute("SELECT COUNT(*) FROM media WHERE jsonpath IS NULL").fetchone()[0]
        total_ts_missing = connection.execute("SELECT COUNT(*) FROM media WHERE chosen_timestamp IS NULL").fetchone()[0]
        total_unhashed_images = connection.execute("SELECT COUNT(*) FROM media WHERE phash IS NULL AND dhash IS NULL").fetchone()[0]
        total_unhashed_videos = connection.execute("SELECT COUNT(*) FROM media WHERE video_hash IS NULL").fetchone()[0]
        tqdm.write(f"🧾 Summary — missing JSON: {total_json_missing:,} • missing timestamps: {total_ts_missing:,} • unhashed images: {total_unhashed_images:,} • unhashed videos: {total_unhashed_videos:,}")
    except Exception:
        pass

    print(f"\n\n🎉 All done! Organized photos are in:\n   {output_directory} \n")
    total_photos = connection.execute("SELECT COUNT(*) FROM media").fetchone()[0]
    tqdm.write(f"📸 Total cataloged items: {total_photos:,}")
    connection.close()


if __name__ == "__main__":
    main()
