#!/usr/bin/env python3
"""
takeout_setup.py

Orchestrator for the Google Takeout Photo Organizer pipeline.

Steps:
1. Initialize SQLite catalog.
2. Scan the target directory for media files and JSON sidecars.
3. Extract EXIF ModifyDate and JSON photoTakenTime for each file.
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
import clean_empty_folders

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
    # 1) Determine target directory
    if len(sys.argv) > 1:
        target_directory = Path(sys.argv[1]).expanduser()
    else:
        target_directory = Path(os.getenv("TARGET_DIR", "")).expanduser()
        if not target_directory.is_dir():
            target_directory = prompt_for_directory(
                "Enter path to your Google Takeout folder"
            )

    # 2) Determine output directory
    output_directory = prompt_for_directory(
        "Enter path to the folder where organized files will be saved"
    )

    # 3) Initialize database
    connection = catalog.initialize_database()

    # 4) Discover all media files and JSON sidecars
    media_file_paths = extract_meta.find_media_files(target_directory)
    json_file_paths = extract_meta.find_json_files(target_directory)

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

    # 7) Extract timestamps from EXIF and JSON
    for media_file_path in tqdm(media_file_paths, desc="Extracting timestamps", unit="file"):
        modify_date = extract_meta.extract_exif_modify_date(media_file_path)
        if modify_date:
            catalog.update_exif_modify_date(
                connection, media_file_path, modify_date
            )
        json_taken_date = extract_meta.extract_json_taken_date(media_file_path)
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

    # 9) Choose the final timestamp for each entry
    choose_timestamp.choose_timestamp_for_all(connection)

    # 10) Rename or move files based on the chosen timestamp
    reorganize.reorganize_files(connection, output_directory)

    # 11) Optional cleanup of empty folders
    if input("🧹 Remove empty folders? [y/N] ").strip().lower().startswith("y"):
        clean_empty_folders(target_directory)

    print(f"\n🎉 All done! Organized photos are in:\n   {output_directory}")


if __name__ == "__main__":
    main()
