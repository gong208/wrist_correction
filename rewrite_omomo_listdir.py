#!/usr/bin/env python3
"""
Simple script to rewrite the entire omomo_listdir.npy file.

This script completely replaces the content of omomo_listdir.npy with new data.
It does not append - it overwrites the entire file.
"""

import numpy as np
import argparse
import os
from typing import List

def rewrite_omomo_listdir(new_content: List[str], file_path: str = 'omomo_listdir.npy') -> bool:
    """
    Rewrite the entire omomo_listdir.npy file with new content.
    
    Args:
        new_content: List of strings to write to the file
        file_path: Path to the .npy file (default: omomo_listdir.npy)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert list to numpy array with object dtype
        data = np.array(new_content, dtype=object)
        
        # Save the data, overwriting the entire file
        np.save(file_path, data)
        
        print(f"Successfully rewrote {file_path} with {len(new_content)} elements")
        print(f"New content: {new_content}")
        return True
        
    except Exception as e:
        print(f"Error rewriting file: {e}")
        return False

def read_current_content(file_path: str = 'omomo_listdir.npy') -> List[str]:
    """
    Read the current content of the file.
    
    Args:
        file_path: Path to the .npy file
    
    Returns:
        List of strings from the file
    """
    try:
        if os.path.exists(file_path):
            data = np.load(file_path, allow_pickle=True)
            return data.tolist()
        else:
            print(f"File {file_path} does not exist")
            return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Rewrite omomo_listdir.npy file with new content')
    parser.add_argument('--file', '-f', default='omomo_listdir.npy', 
                       help='Path to the .npy file (default: omomo_listdir.npy)')
    parser.add_argument('--content', '-c', nargs='+', 
                       help='New content to write to the file')
    parser.add_argument('--read', '-r', action='store_true',
                       help='Read and display current content')
    parser.add_argument('--clear', action='store_true',
                       help='Clear the file (write empty array)')
    parser.add_argument('--example', action='store_true',
                       help='Write example content to the file')
    
    args = parser.parse_args()
    
    if args.read:
        # Read and display current content
        current_content = read_current_content(args.file)
        if current_content:
            print(f"Current content of {args.file}:")
            for i, item in enumerate(current_content):
                print(f"  {i}: {item}")
        else:
            print(f"No content found in {args.file}")
    
    elif args.clear:
        # Clear the file
        rewrite_omomo_listdir([], args.file)
    
    elif args.example:
        # Write example content
        example_content = [
            'sub5_plasticbox_028',
            'sub9_clothesstand_052',
            'sub9_clothesstand_056',
            'sub9_smalltable_021', 
            'sub9_smalltable_045',
            'sub9_smalltable_080',
            'sub9_tripod_028',
            'sub14_monitor_001'
        ]
        rewrite_omomo_listdir(example_content, args.file)
    
    elif args.content:
        # Write specified content
        rewrite_omomo_listdir(args.content, args.file)
    
    else:
        # Interactive mode
        print("Interactive mode - enter new content for omomo_listdir.npy")
        print("Enter one item per line. Press Enter twice to finish:")
        
        new_content = []
        while True:
            try:
                line = input("> ").strip()
                if line == "":
                    break
                new_content.append(line)
            except EOFError:
                break
        
        if new_content:
            rewrite_omomo_listdir(new_content, args.file)
        else:
            print("No content provided")

if __name__ == '__main__':
    main() 