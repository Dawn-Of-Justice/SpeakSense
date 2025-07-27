#!/usr/bin/env python3
"""
Startup script for SpeakSense backend - handles imports correctly
"""

import sys
import os
from pathlib import Path

# Add the backend/src directory to Python path
backend_src = Path(__file__).parent / "src"
sys.path.insert(0, str(backend_src.absolute()))

# Now we can import with the correct module structure
if __name__ == "__main__":
    print("🚀 Starting SpeakSense Backend Server...")
    print(f"📁 Backend source path: {backend_src.absolute()}")
    
    # Set environment variables
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    try:
        # Import the FastAPI app
        from api.fastapi_websocket_server import app
        import uvicorn
        
        print("🌐 Server will be available at:")
        print("   - Backend API: http://localhost:8765")
        print("   - WebSocket: ws://localhost:8765/ws")
        print("\n💡 Press Ctrl+C to stop the server")
        
        # Run the server
        uvicorn.run(
            app,
            host="localhost",
            port=8765,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 This usually means:")
        print("1. Some dependencies might be missing")
        print("2. Model files might not be in the correct location")
        print("3. Path configuration issues")
        
        print(f"\n📝 Python path: {sys.path}")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Backend src exists: {backend_src.exists()}")
        
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
