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
    
    # Store original directory
    original_dir = os.getcwd()
    backend_dir = Path("backend").resolve()
    
    try:
        os.chdir(backend_dir)
        
        # Remove existing venv if it's incomplete and recreate
        venv_path = Path("venv")
        if venv_path.exists():
            import shutil
            print("🗑️ Removing incomplete virtual environment...")
            shutil.rmtree(venv_path)
        
        # Create virtual environment
        run_command([sys.executable, "-m", "venv", "venv"], "Creating virtual environment")
        
        # Verify virtual environment was created properly
        if os.name == 'nt':  # Windows
            pip_executable = venv_path / "Scripts" / "pip.exe"
            python_executable = venv_path / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            pip_executable = venv_path / "bin" / "pip"
            python_executable = venv_path / "bin" / "python"
        
        if not pip_executable.exists():
            raise Exception(f"Virtual environment creation failed - {pip_executable} not found")
        
        # Install requirements
        run_command([str(pip_executable), "install", "-r", "requirements.txt"], "Installing Python dependencies")
        run_command([str(python_executable), "-m", "spacy", "download", "en_core_web_sm"], "Downloading spaCy model")
        
        # Create necessary directories
        dirs = ["logs", "data/temp"]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    finally:
        os.chdir(original_dir)
    
    print("✅ Backend setup completed")

def setup_frontend():
    """Set up the frontend environment"""
    print("🔧 Setting up frontend...")
    
    # Store original directory
    original_dir = os.getcwd()
    frontend_dir = Path("frontend").resolve()
    
    try:
        os.chdir(frontend_dir)
        
        # Use shell=True for Windows compatibility with npm
        if os.name == 'nt':  # Windows
            run_command("npm install", "Installing Node.js dependencies")
        else:  # Unix/Linux/macOS
            run_command(["npm", "install"], "Installing Node.js dependencies")
    finally:
        os.chdir(original_dir)
    
    print("✅ Frontend setup completed")

def start_backend():
    """Start the backend server"""
    print("🚀 Starting backend server...")
    
    # Store original directory
    original_dir = os.getcwd()
    backend_dir = Path("backend").resolve()
    
    try:
        os.chdir(backend_dir)
        
        if os.name == 'nt':  # Windows
            python_executable = Path("venv") / "Scripts" / "python.exe"
        else:  # Unix/Linux/macOS
            python_executable = Path("venv") / "bin" / "python"
        
        if not python_executable.exists():
            print("❌ Virtual environment not found. Please run: python manage.py --setup-backend")
            return
        
        # Use the new server runner
        subprocess.run([str(python_executable), "run_server.py"])
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    finally:
        os.chdir(original_dir)

def start_frontend():
    """Start the frontend development server"""
    print("🚀 Starting frontend development server...")
    
    # Store original directory
    original_dir = os.getcwd()
    frontend_dir = Path("frontend").resolve()
    
    try:
        os.chdir(frontend_dir)
        
        # Use shell=True for Windows compatibility with npm
        if os.name == 'nt':  # Windows
            subprocess.run("npm run dev", shell=True)
        else:  # Unix/Linux/macOS
            subprocess.run(["npm", "run", "dev"])
    except KeyboardInterrupt:
        print("\n🛑 Frontend server stopped")
    finally:
        os.chdir(original_dir)

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
