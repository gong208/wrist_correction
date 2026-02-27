#!/usr/bin/env python3
"""
Script to copy sequences from source to destination without overwriting existing ones
"""

import os
import shutil
from pathlib import Path

def copy_sequences():
    source_dir = Path("/media/volume/Sirui-2/interactcodes/gt/omomo/sequences_canonical")
    dest_dir = Path("/media/volume/Sirui-2/correction/data/omomo/sequences_canonical")
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories from source
    source_subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    print(f"Found {len(source_subdirs)} sequences in source directory")
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for source_subdir in source_subdirs:
        dest_subdir = dest_dir / source_subdir.name
        
        try:
            if dest_subdir.exists():
                print(f"- Skipped {source_subdir.name} (already exists)")
                skipped_count += 1
            else:
                # Copy the entire directory
                shutil.copytree(source_subdir, dest_subdir)
                print(f"✓ Copied {source_subdir.name}")
                copied_count += 1
                
        except Exception as e:
            print(f"✗ Error copying {source_subdir.name}: {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"Copied: {copied_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total source sequences: {len(source_subdirs)}")

if __name__ == "__main__":
    copy_sequences()
