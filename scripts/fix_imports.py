#!/usr/bin/env python3
"""
Fix import paths after reorganization
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import statements in a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix imports based on file location
        if 'backend/src/api/' in str(file_path):
            # API files - use relative imports
            content = re.sub(r'from Live_transcription\.', 'from ..services.transcription.', content)
            content = re.sub(r'from audio_model\.', 'from ..models.audio.', content)
            content = re.sub(r'from LLM', 'from ..services.LLM', content)
            content = re.sub(r'import LIGHT_ASD\.', 'import ..models.asd.', content)
            content = re.sub(r'from LIGHT_ASD\.', 'from ..models.asd.', content)
            
        elif 'backend/src/' in str(file_path) and not 'api' in str(file_path):
            # Main backend files - use relative imports
            content = re.sub(r'from Live_transcription\.', 'from services.transcription.', content)
            content = re.sub(r'from audio_model\.', 'from models.audio.', content)
            content = re.sub(r'from LLM', 'from services.LLM', content)
            content = re.sub(r'import LIGHT_ASD\.', 'import models.asd.', content)
            content = re.sub(r'from LIGHT_ASD\.', 'from models.asd.', content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in: {file_path}")
            return True
        else:
            print(f"ℹ️  No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all imports"""
    backend_src = Path("backend/src")
    
    if not backend_src.exists():
        print("❌ Backend src directory not found!")
        return
    
    python_files = list(backend_src.rglob("*.py"))
    total_fixed = 0
    
    for py_file in python_files:
        if fix_imports_in_file(py_file):
            total_fixed += 1
    
    print(f"\n🎉 Fixed imports in {total_fixed} files")

if __name__ == "__main__":
    main()
