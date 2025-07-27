#!/usr/bin/env python3
"""
SpeakSense Project Management Script
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        if isinstance(command, list):
            result = subprocess.run(command, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"⚠️  {description} completed with warnings")
            if result.stderr.strip():
                print(f"Warnings: {result.stderr.strip()}")
        
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return None

def setup_backend():
    """Set up the backend environment"""
    print("🔧 Setting up backend...")
    
    os.chdir("backend")
    
    # Create virtual environment if it doesn't exist
    if not Path("venv").exists():
        run_command([sys.executable, "-m", "venv", "venv"], "Creating virtual environment")
    
    # Install requirements
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip install -r requirements.txt"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip install -r requirements.txt"
        python_cmd = "venv/bin/python"
    
    run_command(pip_cmd, "Installing Python dependencies")
    run_command(f"{python_cmd} -m spacy download en_core_web_sm", "Downloading spaCy model")
    
    # Create necessary directories
    dirs = ["logs", "data/temp"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    os.chdir("..")
    print("✅ Backend setup completed")

def setup_frontend():
    """Set up the frontend environment"""
    print("🔧 Setting up frontend...")
    
    os.chdir("frontend")
    
    run_command("npm install", "Installing Node.js dependencies")
    
    os.chdir("..")
    print("✅ Frontend setup completed")

def start_backend():
    """Start the backend server"""
    print("🚀 Starting backend server...")
    os.chdir("backend")
    
    if os.name == 'nt':  # Windows
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_cmd = "venv/bin/python"
    
    try:
        # Use the new server runner
        subprocess.run([python_cmd, "run_server.py"])
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    finally:
        os.chdir("..")

def start_frontend():
    """Start the frontend development server"""
    print("🚀 Starting frontend development server...")
    os.chdir("frontend")
    
    try:
        subprocess.run(["npm", "run", "dev"])
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    finally:
        os.chdir("..")

def docker_setup():
    """Set up and run with Docker"""
    print("🐳 Setting up with Docker...")
    
    # Build and run with docker-compose
    run_command("docker-compose build", "Building Docker images")
    run_command("docker-compose up -d", "Starting services with Docker")
    
    print("🎉 Docker setup completed!")
    print("Backend available at: http://localhost:8000")
    print("Frontend available at: http://localhost:3000")

def clean_project():
    """Clean temporary files and caches"""
    print("🧹 Cleaning project...")
    
    patterns_to_remove = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/node_modules",
        "**/.next",
        "**/dist",
        "**/build",
        "**/*.log",
        "**/temp*"
    ]
    
    for pattern in patterns_to_remove:
        for path in Path(".").rglob(pattern.split("/")[-1]):
            if path.exists():
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                    print(f"🗑️  Removed directory: {path}")
                else:
                    path.unlink()
                    print(f"🗑️  Removed file: {path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SpeakSense Project Management")
    parser.add_argument("--setup", action="store_true", help="Set up both backend and frontend")
    parser.add_argument("--setup-backend", action="store_true", help="Set up backend only")
    parser.add_argument("--setup-frontend", action="store_true", help="Set up frontend only")
    parser.add_argument("--start-backend", action="store_true", help="Start backend server")
    parser.add_argument("--start-frontend", action="store_true", help="Start frontend server")
    parser.add_argument("--docker", action="store_true", help="Set up and run with Docker")
    parser.add_argument("--clean", action="store_true", help="Clean temporary files")
    
    args = parser.parse_args()
    
    if args.setup or args.setup_backend:
        setup_backend()
    
    if args.setup or args.setup_frontend:
        setup_frontend()
    
    if args.start_backend:
        start_backend()
    
    if args.start_frontend:
        start_frontend()
    
    if args.docker:
        docker_setup()
    
    if args.clean:
        clean_project()
    
    if not any(vars(args).values()):
        parser.print_help()
        print("\n🚀 Quick start:")
        print("  python manage.py --setup          # Set up both backend and frontend")
        print("  python manage.py --start-backend  # Start backend server")
        print("  python manage.py --start-frontend # Start frontend server")
        print("  python manage.py --docker         # Use Docker setup")

if __name__ == "__main__":
    main()
