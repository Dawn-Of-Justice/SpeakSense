#!/usr/bin/env python3
"""
Setup script for the SpeakSense backend
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return None

def main():
    """Main setup function"""
    print("🚀 Setting up SpeakSense Backend...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    # Create virtual environment
    run_command("python -m venv venv", "Creating virtual environment")
    
    # Activate virtual environment and install requirements
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate && pip install -r requirements.txt"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
    
    run_command(activate_cmd, "Installing Python dependencies")
    
    # Download spaCy model
    spacy_cmd = "python -m spacy download en_core_web_sm"
    if os.name == 'nt':
        spacy_cmd = f"venv\\Scripts\\activate && {spacy_cmd}"
    else:
        spacy_cmd = f"source venv/bin/activate && {spacy_cmd}"
    
    run_command(spacy_cmd, "Downloading spaCy English model")
    
    # Create directories if they don't exist
    dirs = ["logs", "config", "data/temp"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
    
    # Copy environment file if it doesn't exist
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("📝 Created .env file from .env.example")
        else:
            print("⚠️  No .env.example found. Please create a .env file manually.")
    
    print("🎉 Backend setup completed!")
    print("\n📋 Next steps:")
    print("1. Update the .env file with your API keys and configurations")
    print("2. Run 'python src/main.py' to start the main application")
    print("3. Or run 'python src/api/fastapi_websocket_server.py' for the web server")

if __name__ == "__main__":
    main()
