#!/usr/bin/env python3
"""
Data processing utilities for SpeakSense
"""

import os
import shutil
import argparse
from pathlib import Path

def organize_data():
    """Organize data files into proper structure"""
    data_dir = Path("../data")
    
    # Create structure
    (data_dir / "raw" / "audio").mkdir(parents=True, exist_ok=True)
    (data_dir / "raw" / "video").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "features").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "labels").mkdir(parents=True, exist_ok=True)
    
    print("📁 Data directory structure created")

def clean_temp_files():
    """Clean temporary and cache files"""
    patterns = ["__pycache__", "*.pyc", "*.pyo", "*.log", "temp*"]
    
    for pattern in patterns:
        for file_path in Path(".").rglob(pattern):
            if file_path.is_dir():
                shutil.rmtree(file_path)
                print(f"🗑️  Removed directory: {file_path}")
            else:
                file_path.unlink()
                print(f"🗑️  Removed file: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="SpeakSense data utilities")
    parser.add_argument("--organize", action="store_true", help="Organize data structure")
    parser.add_argument("--clean", action="store_true", help="Clean temporary files")
    
    args = parser.parse_args()
    
    if args.organize:
        organize_data()
    
    if args.clean:
        clean_temp_files()
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
