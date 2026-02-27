#!/usr/bin/env python3
"""
Script to replace human.npz with human_fixed.npz in all folders under sequences_canonical
"""

import os
import shutil
from pathlib import Path

def replace_human_files():
    base_dir = Path("/media/volume/Sirui-2/correction/data/omomo/sequences_canonical")
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # Get all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories")
    
    success_count = 0
    error_count = 0
    
    for subdir in subdirs:
        human_npz = subdir / "human.npz"
        human_fixed_npz = subdir / "human_fixed.npz"
        
        try:
            # Check if both files exist
            if human_npz.exists() and human_fixed_npz.exists():
                # Remove the old human.npz file
                human_npz.unlink()
                print(f"✓ Removed {human_npz}")
                
                # Rename human_fixed.npz to human.npz
                human_fixed_npz.rename(human_npz)
                print(f"✓ Renamed {human_fixed_npz} to {human_npz}")
                
                success_count += 1
            elif human_fixed_npz.exists() and not human_npz.exists():
                # Only human_fixed.npz exists, rename it to human.npz
                human_fixed_npz.rename(human_npz)
                print(f"✓ Renamed {human_fixed_npz} to {human_npz}")
                success_count += 1
            elif human_npz.exists() and not human_fixed_npz.exists():
                # Only human.npz exists, no action needed
                print(f"- {subdir.name}: Only human.npz exists, no replacement needed")
                success_count += 1
            else:
                print(f"⚠ {subdir.name}: Neither human.npz nor human_fixed.npz found")
                error_count += 1
                
        except Exception as e:
            print(f"✗ Error processing {subdir.name}: {e}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Total directories: {len(subdirs)}")

if __name__ == "__main__":
    replace_human_files()
