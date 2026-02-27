#!/usr/bin/env python3
"""
Script to copy updated sequences from data/omomo/sequences_canonical 
to ../interactcodes/gt/omomo/sequences_canonical

This will replace only the sequences that exist in the source directory
while keeping all other sequences in the destination unchanged.
"""

import os
import shutil
from pathlib import Path

# Define paths
SOURCE_DIR = Path("/media/volume/Sirui-2/correction/data/omomo/sequences_canonical")
DEST_DIR = Path("/media/volume/Sirui-2/interactcodes/gt/omomo/sequences_canonical")

def main():
    # Check if directories exist
    if not SOURCE_DIR.exists():
        print(f"❌ Source directory does not exist: {SOURCE_DIR}")
        return
    
    if not DEST_DIR.exists():
        print(f"❌ Destination directory does not exist: {DEST_DIR}")
        return
    
    # Get list of sequences to copy
    source_sequences = sorted([d for d in os.listdir(SOURCE_DIR) 
                              if os.path.isdir(SOURCE_DIR / d)])
    
    print(f"Found {len(source_sequences)} sequences in source directory")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print()
    
    # Statistics
    copied = 0
    replaced = 0
    errors = 0
    
    # Copy each sequence
    for i, seq_name in enumerate(source_sequences, 1):
        source_path = SOURCE_DIR / seq_name
        dest_path = DEST_DIR / seq_name
        
        try:
            # Check if sequence already exists in destination
            if dest_path.exists():
                print(f"[{i}/{len(source_sequences)}] Replacing: {seq_name}")
                # Remove old version
                shutil.rmtree(dest_path)
                replaced += 1
            else:
                print(f"[{i}/{len(source_sequences)}] Adding new: {seq_name}")
            
            # Copy the sequence
            shutil.copytree(source_path, dest_path)
            copied += 1
            
        except Exception as e:
            print(f"❌ Error copying {seq_name}: {e}")
            errors += 1
    
    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sequences processed: {len(source_sequences)}")
    print(f"Successfully copied: {copied}")
    print(f"Replaced existing: {replaced}")
    print(f"New sequences added: {copied - replaced}")
    print(f"Errors: {errors}")
    print()
    
    # Verify final count
    dest_sequences = [d for d in os.listdir(DEST_DIR) 
                      if os.path.isdir(DEST_DIR / d)]
    print(f"Total sequences in destination after copy: {len(dest_sequences)}")
    print()

if __name__ == "__main__":
    main()

