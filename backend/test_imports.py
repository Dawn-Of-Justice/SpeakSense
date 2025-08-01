#!/usr/bin/env python3
"""
Simple test to see what's wrong with imports
"""

import sys
import os
from pathlib import Path

# Add the backend/src directory to Python path
backend_src = Path(__file__).parent / "src"
sys.path.insert(0, str(backend_src.absolute()))

print("🔍 Testing imports...")
print(f"📁 Backend source path: {backend_src.absolute()}")

try:
    print("✓ Testing services.LLM...")
    from services.LLM import AI
    print("✅ LLM import successful!")
    
except Exception as e:
    print(f"❌ LLM import failed: {e}")
    import traceback
    traceback.print_exc()
    
try:
    print("✓ Testing FastAPI...")
    from api.fastapi_websocket_server import app
    print("✅ FastAPI import successful!")
    
except Exception as e:
    print(f"❌ FastAPI import failed: {e}")
    import traceback
    traceback.print_exc()
