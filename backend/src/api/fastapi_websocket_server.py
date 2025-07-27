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
import torch

# Set environment variables for PyTorch optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def patch_s3fd_metatensor_fix():
    """Apply meta tensor fix specifically for S3FD model"""
    try:
        # Patch the global .to() method for meta tensors
        original_to = torch.nn.Module.to
        
        def fixed_to(self, *args, **kwargs):
            """Fixed .to() method that handles meta tensors"""
            try:
                # Check if any parameters are meta tensors
                has_meta = any(p.is_meta for p in self.parameters() if hasattr(p, 'is_meta'))
                
                if has_meta:
                    print("Warning: Found meta tensors, using to_empty() instead")
                    device = args[0] if args else kwargs.get('device', 'cpu')
                    dtype = kwargs.get('dtype', None)
                    
                    # Use to_empty for meta tensors
                    try:
                        empty_module = self.to_empty(device=device)
                        if dtype is not None:
                            empty_module = empty_module.type(dtype)
                        return empty_module
                    except:
                        # Fallback: force CPU
                        print("to_empty() failed, using CPU fallback")
                        return original_to(self, 'cpu')
                else:
                    return original_to(self, *args, **kwargs)
                    
            except Exception as e:
                print(f"Fixed .to() failed, using CPU fallback: {e}")
                return original_to(self, 'cpu')
        
        # Apply the patch
        torch.nn.Module.to = fixed_to
        print("Meta tensor fix applied successfully")
        return True
        
    except Exception as e:
        print(f"Failed to apply meta tensor fix: {e}")
        return False

nlp = spacy.blank("en")  # Load English tokenizer

def remove_non_english_text(response):
    doc = nlp(response)
    return ' '.join([token.text for token in doc if token.is_alpha])  # Keep only alphabetic words

# Global variables - same as original design
transcription_queue = queue.Queue()
response_queue = queue.Queue()
video_frame_queue = queue.Queue()  # Add video frame queue
websocket_broadcast_queue = None  # Will be initialized as asyncio.Queue in lifespan
clear_state = False
transcribed_stuff = None
last_sent_transcription = None  # Track last sent transcription to avoid duplicates
last_transcription_time = 0  # Track when last transcription was sent
ai_is_speaking = False
transcription_active = False  # Control transcription
system_started = False  # Control overall system

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
    # Put message in async queue to be processed by the WebSocket handler
    if websocket_broadcast_queue is not None:
        try:
            # Use put_nowait to avoid blocking the calling thread
            websocket_broadcast_queue.put_nowait(message)
        except asyncio.QueueFull:
            print(f"WebSocket queue full, dropping message: {message.get('type', 'unknown')}")
    else:
        print(f"WebSocket queue not initialized, dropping message: {message.get('type', 'unknown')}")

async def websocket_message_processor():
    """Background task to process WebSocket messages from the queue"""
    while True:
        try:
            # Wait for messages from the queue
            message = await websocket_broadcast_queue.get()
            
            # Send to all connected clients
            with websocket_lock:
                disconnected_clients = []
                for client in websocket_clients:
                    try:
                        await client.send_text(json.dumps(message))
                    except Exception as e:
                        print(f"WebSocket send error: {e}")
                        disconnected_clients.append(client)
                
                # Remove disconnected clients
                for client in disconnected_clients:
                    if client in websocket_clients:
                        websocket_clients.remove(client)
            
            # Mark task as done
            websocket_broadcast_queue.task_done()
            
        except Exception as e:
            print(f"WebSocket message processor error: {e}")
            await asyncio.sleep(0.1)

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
                    broadcast_to_clients(message)
                    
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            print(f"Video streaming thread error: {e}")
            time.sleep(0.1)

def transcription_thread():
    """Handles real-time transcription and puts results in a queue"""
    global clear_state, transcribed_stuff, ai_is_speaking, transcription_active, last_sent_transcription, last_transcription_time
    
    transcriber = WhisperRealtimeTranscriber()
    last_processed_length = 0  # Track how much of the transcription we've already processed
    
    try:
        # Wait for activation
        print("Transcription thread waiting for activation...")
        while not transcription_active:
            time.sleep(0.1)
            
        print("Transcription thread activated! Starting transcriber...")
        # Start listening and transcribing
        transcriber.start_listening()
        transcriber.start_transcribing()
        print("Transcriber started successfully!")
        
        # Keep running until stopped
        while transcription_active:
            with lock:
                # Only process transcription when AI is not speaking
                if not ai_is_speaking:
                    if clear_state:
                        transcriber.last_transcription = ""
                        last_sent_transcription = None  # Reset last sent transcription
                        last_transcription_time = 0
                        last_processed_length = 0  # Reset processed length
                        clear_state = False
                        print("Cleared transcription state")
                    
                    # Check if there's a new transcription
                    if hasattr(transcriber, 'last_transcription') and transcriber.last_transcription:
                        current_transcription = transcriber.last_transcription
                        current_length = len(current_transcription)
                        
                        # Check if there's new content since last processing
                        if current_length > last_processed_length:
                            new_content = current_transcription[last_processed_length:]
                            print(f"Raw transcription received: '{current_transcription}'")
                            print(f"New content: '{new_content}'")
                            
                            # Remove hallucinated words from the full transcription
                            hallucinated_words = ["See you next time", "!", "Thank you for watching", "thank you", "?"]
                            cleaned_transcription = current_transcription
                            for word in hallucinated_words:
                                if word in cleaned_transcription:
                                    cleaned_transcription = cleaned_transcription.replace(word, "")
                            
                            transcribed_stuff = cleaned_transcription
                            current_time = time.time()
                            
                            # Send if there's meaningful new content
                            should_send = False
                            
                            if last_sent_transcription is None:
                                # First transcription
                                should_send = True
                                print("First transcription - will send")
                            elif cleaned_transcription != last_sent_transcription:
                                # Text has changed
                                should_send = True
                                print("Transcription changed - will send")
                            elif current_time - last_transcription_time > 3.0:
                                # Force send after 3 seconds (user might be continuing)
                                should_send = True
                                print("Force sending after 3 seconds")
                            
                            if should_send and websocket_clients:
                                broadcast_to_clients({
                                    "type": "transcription",
                                    "text": transcribed_stuff,
                                    "timestamp": current_time
                                })
                                last_sent_transcription = transcribed_stuff
                                last_transcription_time = current_time
                                last_processed_length = current_length
                                print(f"✅ Sent transcription: '{transcribed_stuff}'")
                            elif transcribed_stuff:
                                print(f"⏭️  Skipped duplicate transcription: '{transcribed_stuff}'")
                        else:
                            # Print periodically to show the thread is running
                            if int(time.time()) % 5 == 0:
                                print("Transcription thread running, waiting for new audio...")
                    else:
                        # Print periodically to show the thread is running
                        if int(time.time()) % 10 == 0:
                            print("Transcription thread running, waiting for audio...")
                            
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        transcriber.stop()

def addressing_thread():
    """Processes transcriptions and determines if AI should respond"""
    global clear_state, transcribed_stuff, system_started
    try:
        # Wait for system to start
        while not system_started:
            time.sleep(0.1)
            
        classifier = AddressClassifierPt()
        
        context = ""
        last_processed = time.time()
        process_interval = 0.2  # Process context every 200ms
        
        while True:
            if not system_started:
                time.sleep(0.1)
                continue
                
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
                    broadcast_to_clients({
                        "type": "addressing_status", 
                        "is_addressing": current_shared_state,
                        "timestamp": time.time()
                    })
                
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
            broadcast_to_clients({
                "type": "ai_speaking",
                "is_speaking": True,
                "timestamp": time.time()
            })

        response = chat.generate_response(
            prompt=f"Input: {prompt_text}", 
            system_message="You are an AI model who gives replies to crippled conversation try your best to figure out what the user meant from the cut off or maybe not so cut of user query, reply in 2 sentences"
        )
        cleaned_response = remove_non_english_text(response)

        # Send AI response to WebSocket clients
        if websocket_clients:
            broadcast_to_clients({
                "type": "ai_response",
                "text": cleaned_response,
                "timestamp": time.time()
            })

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
            broadcast_to_clients({
                "type": "ai_speaking",
                "is_speaking": False,
                "timestamp": time.time()
            })

def asd_thread():
    """Runs the Active Speaker Detection in a separate thread"""
    try:
        # Apply meta tensor fix before starting ASD
        patch_s3fd_metatensor_fix()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Starting ASD with meta tensor fixes...")
        # Pass the video callback to enable streaming
        asd_main(run_sub_audio_thread=True, video_callback=video_frame_callback)
        
    except Exception as e:
        print(f"ASD thread error: {e}")
        traceback.print_exc()
        
        # Fallback: Force CPU-only mode
        try:
            print("Attempting ASD restart with CPU-only mode...")
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            torch.set_default_tensor_type('torch.FloatTensor')
            time.sleep(1)
            
            # Re-apply the patch
            patch_s3fd_metatensor_fix()
            asd_main(run_sub_audio_thread=True, video_callback=video_frame_callback)
            
        except Exception as restart_error:
            print(f"ASD CPU fallback also failed: {restart_error}")
            print("ASD will not be available")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global websocket_broadcast_queue
    
    # Initialize the async queue
    websocket_broadcast_queue = asyncio.Queue(maxsize=100)
    
    # Apply PyTorch fixes before starting threads
    print("Starting ASD thread...")
    
    # Startup code
    asd = threading.Thread(target=asd_thread)
    transcription = threading.Thread(target=transcription_thread)
    addressing = threading.Thread(target=addressing_thread)
    video_streaming = threading.Thread(target=video_streaming_thread)
    
    # Set threads as daemon
    asd.daemon = True
    transcription.daemon = True
    addressing.daemon = True
    video_streaming.daemon = True
    
    # Start the WebSocket message processor as a background task
    websocket_task = asyncio.create_task(websocket_message_processor())
    
    asd.start()
    time.sleep(1)  # Give ASD time to initialize
    
    print("Starting other threads...")
    transcription.start() 
    addressing.start()
    video_streaming.start()
    
    yield
    
    # Shutdown code
    print("Shutting down server...")
    websocket_task.cancel()
    try:
        await websocket_task
    except asyncio.CancelledError:
        pass

# Update app creation
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_active, system_started
    
    print(f"🔌 WebSocket connection attempt from {websocket.client}")
    await websocket.accept()
    print(f"✅ WebSocket connection accepted from {websocket.client}")
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
                elif data.get("type") == "start_transcription":
                    transcription_active = True
                    system_started = True
                    print("Transcription started via frontend")
                elif data.get("type") == "stop_transcription":
                    transcription_active = False
                    print("Transcription stopped via frontend")
                    
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
        "transcription_active": transcription_active,
        "system_started": system_started,
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