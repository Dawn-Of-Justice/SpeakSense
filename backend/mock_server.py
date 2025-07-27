#!/usr/bin/env python3
"""
Simplified SpeakSense backend server for testing
This version starts without loading heavy ML models
"""

import sys
import os
from pathlib import Path
import asyncio
import json
import time
import queue
import threading

# Add the backend/src directory to Python path
backend_src = Path(__file__).parent / "src"
sys.path.insert(0, str(backend_src.absolute()))

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Simple mock classes for testing
class MockTranscriber:
    def __init__(self):
        self.last_transcription = ""
        self.running = False
    
    def start_listening(self):
        self.running = True
        print("🎤 Mock transcriber listening...")
    
    def start_transcribing(self):
        # Simulate some transcription
        def mock_transcription():
            time.sleep(2)
            if self.running:
                self.last_transcription = "Hello, this is a test transcription"
                time.sleep(3)
                self.last_transcription = "How are you doing today?"
        
        threading.Thread(target=mock_transcription, daemon=True).start()
    
    def stop(self):
        self.running = False

class MockClassifier:
    def classify_text(self, text):
        # Mock classification - always return True for testing
        return {
            "is_addressing_robot": True,
            "confidence": 0.85
        }

class MockAI:
    def generate_response(self, prompt, system_message=None):
        return f"Mock AI response to: {prompt[:50]}..."

# Global variables
transcription_queue = queue.Queue()
response_queue = queue.Queue()
clear_state = False
transcribed_stuff = None
ai_is_speaking = False

# Mock components
transcriber = MockTranscriber()
classifier = MockClassifier()
ai_chat = MockAI()

# WebSocket clients
websocket_clients = []
websocket_lock = threading.Lock()

def add_websocket_client(websocket):
    with websocket_lock:
        websocket_clients.append(websocket)

def remove_websocket_client(websocket):
    with websocket_lock:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

async def broadcast_to_clients(message):
    """Send message to all connected WebSocket clients"""
    if not websocket_clients:
        return
        
    disconnected_clients = []
    for client in websocket_clients:
        try:
            await client.send_text(json.dumps(message))
        except Exception as e:
            print(f"WebSocket send error: {e}")
            disconnected_clients.append(client)
    
    # Remove disconnected clients
    for client in disconnected_clients:
        with websocket_lock:
            if client in websocket_clients:
                websocket_clients.remove(client)

def transcription_thread():
    """Mock transcription thread"""
    global transcribed_stuff
    
    transcriber.start_listening()
    transcriber.start_transcribing()
    
    while True:
        if transcriber.last_transcription and transcriber.last_transcription != transcribed_stuff:
            transcribed_stuff = transcriber.last_transcription
            print(f"📝 Transcription: {transcribed_stuff}")
            
            # Send to WebSocket clients using thread-safe approach
            if websocket_clients:
                try:
                    # Get the main event loop and schedule the coroutine
                    loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        broadcast_to_clients({
                            "type": "transcription",
                            "text": transcribed_stuff,
                            "timestamp": time.time()
                        }),
                        loop
                    )
                except RuntimeError:
                    print(f"Could not send transcription via WebSocket: {transcribed_stuff}")
        
        time.sleep(0.5)

def addressing_thread():
    """Mock addressing detection thread"""
    global transcribed_stuff, ai_is_speaking
    
    while True:
        if transcribed_stuff and not ai_is_speaking:
            result = classifier.classify_text(transcribed_stuff)
            print(f"🤖 Classification: {result}")
            
            if result["is_addressing_robot"] and result["confidence"] > 0.6:
                print(f"🎯 Generating response for: {transcribed_stuff}")
                
                # Generate AI response
                ai_is_speaking = True
                response = ai_chat.generate_response(transcribed_stuff)
                print(f"💬 AI Response: {response}")
                
                # Send response to WebSocket clients  
                if websocket_clients:
                    try:
                        loop = asyncio.get_event_loop()
                        asyncio.run_coroutine_threadsafe(
                            broadcast_to_clients({
                                "type": "ai_response", 
                                "text": response,
                                "timestamp": time.time()
                            }),
                            loop
                        )
                    except RuntimeError:
                        print("Could not send AI response via WebSocket")
                
                # Reset state
                time.sleep(2)  # Simulate speaking time
                ai_is_speaking = False
                transcribed_stuff = None
        
        time.sleep(1)

# FastAPI app
app = FastAPI(title="SpeakSense Mock API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    add_websocket_client(websocket)
    print(f"👥 WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    try:
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except WebSocketDisconnect:
                break
            except Exception as e:
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        remove_websocket_client(websocket)
        print(f"👥 WebSocket client disconnected. Total clients: {len(websocket_clients)}")

@app.get("/")
async def root():
    return {"message": "SpeakSense Mock Server", "status": "running"}

@app.get("/status")
async def status():
    return {
        "clients": len(websocket_clients),
        "ai_speaking": ai_is_speaking,
        "transcribed_text": transcribed_stuff,
        "mode": "mock"
    }

if __name__ == "__main__":
    print("🚀 Starting SpeakSense Mock Backend Server...")
    print(f"📁 Backend source path: {backend_src.absolute()}")
    print("🌐 Server will be available at:")
    print("   - Backend API: http://localhost:8765")
    print("   - WebSocket: ws://localhost:8765/ws")
    print("   - Status: http://localhost:8765/status")
    print("\n💡 This is a MOCK version for testing the frontend")
    print("💡 Press Ctrl+C to stop the server")
    
    # Start background threads
    threading.Thread(target=transcription_thread, daemon=True).start()
    threading.Thread(target=addressing_thread, daemon=True).start()
    
    # Run the server
    uvicorn.run(
        app,
        host="localhost",
        port=8765,
        log_level="info"
    )
