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

import os
import sys
from pathlib import Path

# Import project modules
import catalog
import extract_meta
import choose_timestamp
import reorganize
from clean_empty_folders import clean_empty_folders
import dedupe 

from tqdm import tqdm


def prompt_for_directory(prompt_message: str) -> Path:
    """Prompt the user to enter a directory path until a valid one is provided."""
    while True:
        user_input = input(f"{prompt_message}: ").strip()
        directory = Path(user_input).expanduser()
        if directory.is_dir():
            return directory
        print(f"⚠️  '{user_input}' is not a valid directory. Please try again.")


def main():
    """# 1) Determine target directory
    target_directory = prompt_for_directory(
        "Enter path to your Google Takeout folder"
    )

    # 2) Determine output directory
    output_directory = prompt_for_directory(
        "Enter path to the folder where organized files will be saved"
    )"""

    target_directory = Path('/Users/kamil1/Pictures/teste').expanduser()
    output_directory = Path('/Users/kamil1/Pictures/teste/final').expanduser()
    
    # 3) Initialize database
    print(f"\nCreating database")
    connection = catalog.initialize_database()

    # 4) Discover media files with progress bar
    print("\nScanning for media files...")
    media_iterator = extract_meta.find_media_files(target_directory)
    media_file_paths = list(tqdm(media_iterator, desc="Finding media files", unit="file"))
    print(f"✅ Found {len(media_file_paths)} media files")

    # 4b) Discover JSON sidecars with progress bar
    print("\nScanning for JSON sidecars...")
    json_iterator = extract_meta.find_json_files(target_directory)
    json_file_paths = list(tqdm(json_iterator, desc="Finding JSON files", unit="file"))
    print(f"✅ Found {len(json_file_paths)} JSON files\n")

    # Correct any mismatched extensions in place
    corrected_media_paths = []
    for media_file_path in tqdm(media_file_paths, desc="Correcting extensions", unit="file"):
        corrected = extract_meta.correct_file_extension_by_mime(media_file_path)
        corrected_media_paths.append(corrected)
    media_file_paths = corrected_media_paths

    # 5) Populate catalog entries
    for media_file_path in tqdm(media_file_paths, desc="Populating catalog", unit="file"):
        catalog.add_media_entry(connection, media_file_path)

    # 6) Associate JSON sidecars
    for media_file_path in tqdm(media_file_paths, desc="Associating JSON", unit="file"):
        matching_json = extract_meta.match_json_for_media(
            media_file_path, json_file_paths
        )
        if matching_json:
            catalog.update_json_path(
                connection, media_file_path, matching_json
            )

    print(f"\n")
    
    # 7) Extract timestamps from EXIF and JSON
    for media_file_path in tqdm(media_file_paths, desc="Extracting timestamps", unit="file"):
        create_date = extract_meta.extract_exif_create_date(media_file_path)
        if create_date:
            catalog.update_exif_create_date(
                connection, media_file_path, create_date
            )
        json_taken_date = extract_meta.extract_json_taken_date(media_file_path, json_file_paths)
        if json_taken_date:
            catalog.update_json_taken_date(
                connection, media_file_path, json_taken_date
            )

    # 8) Fallback: parse date from filename
    for media_file_path in tqdm(media_file_paths, desc="Parsing filenames", unit="file"):
        filename_date = extract_meta.parse_date_from_filename(
            media_file_path.stem
        )
        if filename_date:
            catalog.update_filename_parsed_date(
                connection, media_file_path, filename_date
            )

    print(f"\n")

    # 9) Choose the final timestamp for each entry
    for _ in tqdm(range(1), desc="Choosing timestamps", unit="step"):
        choose_timestamp.choose_timestamp_for_all(connection)

    # 10) Compute and store perceptual hashes (pHash & dHash)
    for media_file_path in tqdm(media_file_paths, desc="Hashing images", unit="file"):
        perceptual_hash, difference_hash = dedupe.compute_perceptual_hashes(media_file_path)
        if perceptual_hash:
            catalog.update_phash(connection, media_file_path, perceptual_hash)
        if difference_hash:
            catalog.update_dhash(connection, media_file_path, difference_hash)

    print(f"\n")

    # 11) Deduplication pass
    print("\n🔍 Running dedupe pass and renaming duplicates in place")
    for _ in tqdm(range(1), desc="\nAnalising files for duplicates", unit="step"):
        dedupe.find_duplicates(connection, output_directory)

    print(f"\n")
    
    # 12) Move files based on the chosen timestamp
    for _ in tqdm(range(1), desc="\nReorganizing files", unit="file"):
        reorganize.reorganize_files(connection, output_directory)

    # 13) Optional cleanup of empty folders
    if input("\n🧹 Remove empty folders? [y/N] ").strip().lower().startswith("y"):
        for _ in tqdm(range(1), desc="Cleaning empty folders", unit="folder"):
            clean_empty_folders(target_directory)


    print(f"\n🎉 All done! Organized photos are in:\n   {output_directory}")


if __name__ == "__main__":
    main()
