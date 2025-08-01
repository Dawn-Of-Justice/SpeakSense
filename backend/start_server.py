#!/usr/bin/env python3
"""
Startup script for SpeakSense backend that handles Python path correctly
"""

import sys
import os
from pathlib import Path

# Add the backend/src directory to Python path
backend_src = Path(__file__).parent / "src"
sys.path.insert(0, str(backend_src.absolute()))

try:
    # Now import and run the FastAPI server
    from api.fastapi_websocket_server import app
    import uvicorn
    
    if __name__ == "__main__":
        print("🚀 Starting SpeakSense Backend Server...")
        print(f"📁 Backend source path: {backend_src.absolute()}")
        print("🌐 Server will be available at:")
        print("   - Backend API: http://localhost:8000")
        print("   - API Documentation: http://localhost:8000/docs")
        print("   - WebSocket: ws://localhost:8000/ws")
        print("\n💡 Press Ctrl+C to stop the server")
        
        uvicorn.run(
            "api.fastapi_websocket_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Trying to fix import paths...")
    
    # Fallback: try to import from the old structure
    try:
        # Add project root to path for old imports
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root.absolute()))
        
        from backend.src.api.fastapi_websocket_server import app
        import uvicorn
        
        print("✅ Using fallback import method")
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
        
    except Exception as e2:
        print(f"❌ Fallback failed: {e2}")
        print("\n🛠️ Please check the following:")
        print("1. All dependencies are installed: pip install -r requirements.txt")
        print("2. spaCy model is downloaded: python -m spacy download en_core_web_sm")
        print("3. File paths are correct")
        sys.exit(1)
