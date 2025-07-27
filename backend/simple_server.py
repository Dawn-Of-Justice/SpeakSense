#!/usr/bin/env python3
"""
Simple FastAPI server for testing the reorganized structure
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI(title="SpeakSense API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "SpeakSense Backend is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend server is operational"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - replace with actual logic later
            await websocket.send_text(f"Received: {data}")
    except WebSocketDisconnect:
        print("WebSocket disconnected")

if __name__ == "__main__":
    print("🚀 Starting SpeakSense Backend Server...")
    print("🌐 Server will be available at:")
    print("   - Backend API: http://localhost:8000")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - WebSocket: ws://localhost:8000/ws")
    print("\n💡 Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
