#!/usr/bin/env python3
"""
clean_empty_folders.py

Recursively deletes any directory that is empty or contains only a .DS_Store file.
"""

from pathlib import Path
from tqdm import tqdm

def clean_empty_folders(target_directory: Path) -> None:
    """
    Walk through all subdirectories of target_directory in reverse order (deepest first).
    Delete a directory if it is empty or contains only a '.DS_Store' file.
    """

    removed_count = 0
    # Gather all directories, sorted deepest-first
    all_directories = sorted(
        [p for p in target_directory.rglob('*') if p.is_dir()],
        key=lambda p: len(str(p)),
        reverse=True
    )

    IGNORABLE = {'.DS_Store', 'metadata.json'}

    for directory in all_directories:
        # Prevent deletion of the root directory itself
        if directory == target_directory:
            continue
        try:
            # Ignore .DS_Store and metadata.json when checking for content
            children = [
                child for child in directory.iterdir()
                if not (child.is_file() and child.name in IGNORABLE)
            ]

            if not children:
                # Remove ignorable files if present
                for name in IGNORABLE:
                    file = directory / name

                    if file.exists():
                        file.unlink()

                # Remove the empty directory
                directory.rmdir()
                removed_count += 1
                print(f"\n🗑 Removed empty folder: {directory}")

        except Exception:
            # Skip any directory we cannot remove
            pass
        
    if removed_count == 0:
        print(f"\n✅ No empty folders found in {target_directory}")