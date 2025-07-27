# fastapi_websocket_server.py
import asyncio
import json
import threading
import time
import queue
import traceback
import sys
import importlib
import base64
import cv2
from pathlib import Path

# Add the src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from services.transcription.Transcription3 import WhisperRealtimeTranscriber
# from models.audio.Classifier import AddressClassifier
from models.audio.Classifier import AddressClassifierPt
from services.LLM import AI
import pyttsx3
from models.asd.realtime3 import main as asd_main
from models.asd import realtime3 as asd_module
import playsound
import spacy
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


nlp = spacy.blank("en")  # Load English tokenizer

def remove_non_english_text(response):
    doc = nlp(response)
    return ' '.join([token.text for token in doc if token.is_alpha])  # Keep only alphabetic words

# Global variables - same as original design
transcription_queue = queue.Queue()
response_queue = queue.Queue()
video_frame_queue = queue.Queue()  # Add video frame queue
clear_state = False
transcribed_stuff = None
ai_is_speaking = False

# Initialize AI components
chat = AI()
lock = threading.Condition()

# WebSocket clients list
websocket_clients = []
websocket_lock = threading.Lock()

def add_websocket_client(websocket):
    with websocket_lock:
        websocket_clients.append(websocket)

def remove_websocket_client(websocket):
    with websocket_lock:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

def broadcast_to_clients(message):
    """Send message to all connected WebSocket clients"""
    with websocket_lock:
        disconnected_clients = []
        for client in websocket_clients:
            try:
                # Use asyncio.run_coroutine_threadsafe for thread safety
                if hasattr(asyncio, '_get_running_loop'):
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(
                            client.send_text(json.dumps(message)), loop
                        )
                    except RuntimeError:
                        # No running loop, create a new one
                        asyncio.run(client.send_text(json.dumps(message)))
                else:
                    asyncio.run(client.send_text(json.dumps(message)))
            except Exception as e:
                print(f"WebSocket send error: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            websocket_clients.remove(client)

def video_frame_callback(frame):
    """Callback function to receive video frames from ASD"""
    try:
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Put in queue for streaming
        video_frame_queue.put(frame_base64)
    except Exception as e:
        print(f"Error in video frame callback: {e}")

def video_streaming_thread():
    """Thread to stream video frames to WebSocket clients"""
    while True:
        try:
            if not video_frame_queue.empty():
                frame_data = video_frame_queue.get()
                
                # Send to all connected clients
                if websocket_clients:
                    message = {
                        "type": "video_frame",
                        "data": frame_data,
                        "timestamp": time.time()
                    }
                    threading.Thread(target=broadcast_to_clients, args=(message,)).start()
                    
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Video streaming thread error: {e}")
            time.sleep(0.1)

def transcription_thread():
    """Handles real-time transcription and puts results in a queue"""
    global clear_state, transcribed_stuff, ai_is_speaking
    
    transcriber = WhisperRealtimeTranscriber()
    
    try:
        # Start listening and transcribing
        transcriber.start_listening()
        transcriber.start_transcribing()
        
        # Keep running until Ctrl+C
        while True:
            with lock:
                # Only process transcription when AI is not speaking
                if not ai_is_speaking:
                    if clear_state:
                        transcriber.last_transcription = ""
                        clear_state = False
                    if transcriber.last_transcription:
                        hallucinated_words = ["See you next time", "!", "Thank you for watching", "thank you", "?"]
                        for word in hallucinated_words:
                            if word in transcriber.last_transcription:
                                transcriber.last_transcription = transcriber.last_transcription.replace(word, "")
                        transcribed_stuff = transcriber.last_transcription
                        
                        # Send transcription to WebSocket clients
                        if websocket_clients:
                            threading.Thread(target=broadcast_to_clients, args=({
                                "type": "transcription",
                                "text": transcribed_stuff,
                                "timestamp": time.time()
                            },)).start()
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        transcriber.stop()

def addressing_thread():
    """Processes transcriptions and determines if AI should respond"""
    global clear_state, transcribed_stuff
    try:
        classifier = AddressClassifierPt()
        
        context = ""
        last_processed = time.time()
        process_interval = 0.2  # Process context every 200ms
        
        while True:
            if transcribed_stuff:
                context = transcribed_stuff
            
            current_time = time.time()
            # Only process at intervals to avoid constant classification
            if current_time - last_processed > process_interval and context.strip():
                # Directly access the module's shared_state for up-to-date value
                current_shared_state = asd_module.shared_state
                print(f"ASD Detection: {current_shared_state}")
                print(f"Transcription: {context}")
                if current_shared_state is None:
                    current_shared_state = False
                
                # Send ASD status to WebSocket clients
                if websocket_clients:
                    threading.Thread(target=broadcast_to_clients, args=({
                        "type": "asd_status",
                        "active": current_shared_state,
                        "timestamp": time.time()
                    },)).start()
                
                if current_shared_state:  # Check if speaker is looking at camera/active
                    try:
                        out = classifier.classify_text(context)
                        if out["is_addressing_robot"] and out['confidence'] > 0.6:
                            print(f"Transcribed Context: {context}")
                            with lock:
                                generate_response(context)
                        # print(f"")
                                
                    except Exception as e:
                        print(f"Error in classification: {e}")
                        traceback.print_exc()
                
                # Reset context if it gets too long
                if len(context) >= 5000:
                    clear_state = True
                    
                last_processed = current_time
            
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in addressing thread: {e}")
        traceback.print_exc()

def generate_response(prompt_text):
    """Generate AI response in a separate thread"""
    global clear_state, ai_is_speaking, transcribed_stuff
    
    try:
        # Set flag that AI is speaking
        with lock:
            ai_is_speaking = True
        
        # Send AI speaking status to WebSocket clients
        if websocket_clients:
            threading.Thread(target=broadcast_to_clients, args=({
                "type": "ai_speaking",
                "speaking": True,
                "timestamp": time.time()
            },)).start()

        response = chat.generate_response(
            prompt=f"Input: {prompt_text}", 
            system_message="You are an AI model who gives replies to crippled conversation try your best to figure out what the user meant from the cut off or maybe not so cut of user query, reply in 2 sentences"
        )
        cleaned_response = remove_non_english_text(response)

        # Send AI response to WebSocket clients
        if websocket_clients:
            threading.Thread(target=broadcast_to_clients, args=({
                "type": "ai_response",
                "text": cleaned_response,
                "timestamp": time.time()
            },)).start()

        engine = pyttsx3.init()
        engine.setProperty("rate", 140)
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[1].id)
        engine.save_to_file(cleaned_response, "sample.wav")
        engine.runAndWait()
        playsound.playsound("./sample.wav")
        
        response_queue.put(cleaned_response)
    finally:
        # Always reset flag when done speaking, even if there was an error
        with lock:
            ai_is_speaking = False
            clear_state = True
            transcribed_stuff = None  # Reset transcribed content after speaking
            
        # Send AI speaking status to WebSocket clients
        if websocket_clients:
            threading.Thread(target=broadcast_to_clients, args=({
                "type": "ai_speaking",
                "speaking": False,
                "timestamp": time.time()
            },)).start()

def asd_thread():
    """Runs the Active Speaker Detection in a separate thread"""
    try:
        # Set device before running ASD
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # Pass the video callback to enable streaming
        asd_main(run_sub_audio_thread=True, video_callback=video_frame_callback)
    except Exception as e:
        print(f"ASD thread error: {e}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    asd = threading.Thread(target=asd_thread)
    transcription = threading.Thread(target=transcription_thread)
    addressing = threading.Thread(target=addressing_thread)
    video_streaming = threading.Thread(target=video_streaming_thread)
    
    transcription.daemon = True
    addressing.daemon = True
    video_streaming.daemon = True
    
    asd.start()
    time.sleep(1)
    transcription.start() 
    addressing.start()
    video_streaming.start()
    
    yield
    # Shutdown code (if needed)

# Update app creation
app = FastAPI(lifespan=lifespan)




# # Create FastAPI app
# app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from contextlib import asynccontextmanager



# @app.on_event("startup")
# async def startup_event():
#     """Initialize background threads when FastAPI starts"""
#     # Create threads - exactly like original design
#     asd = threading.Thread(target=asd_thread)
#     transcription = threading.Thread(target=transcription_thread)
#     addressing = threading.Thread(target=addressing_thread)
    
#     # Set as daemon threads so they exit when main program exits
#     transcription.daemon = True
#     addressing.daemon = True
    
#     # Start threads - exactly like original design
#     asd.start()
#     time.sleep(1)  # Give ASD thread time to initialize
#     transcription.start() 
#     addressing.start()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    add_websocket_client(websocket)
    
    try:
        while True:
            try:
                # Keep connection alive by receiving messages
                message = await websocket.receive_text()
                # Echo back any received messages for testing
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

@app.get("/")
async def root():
    return {"message": "AI Transcription WebSocket Server"}

@app.get("/status")
async def status():
    return {
        "clients": len(websocket_clients),
        "ai_speaking": ai_is_speaking,
        "asd_active": getattr(asd_module, 'shared_state', False)
    }

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_websocket_server:app",
        host="localhost",
        port=8765,
        reload=False,
        log_level="error"  # Minimize terminal output
    )