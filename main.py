import threading
from Live_transcription.Transcription import WhisperRealtimeTranscriber
from Live_transcription.Transcription2 import RealTimeTranscriber
import time

def transcription():
    # Create the Whisper transcriber
    # transcriber = WhisperRealtimeTranscriber()
    transcriber =  RealTimeTranscriber()
    
    try:
        # Start listening and transcribing
        transcriber.start_listening()
        transcriber.start_transcribing()
        
        # Keep running until Ctrl+C
        print("Press Ctrl+C to stop transcription")
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        transcriber.stop()

def ASD():
    # ASD realtime code goes here
    while True:
        
        with open("transcription.txt", "r") as f:
            s = f.read()
            
        if len(s) >= 100:
            with open("transcription.txt", "w") as f:
                f.write("")
        
if __name__ == "__main__":
    transcription_thread = threading.Thread(target=transcription)
    ASD_thread = threading.Thread(target=ASD)
    
    # Threads are getting started
    transcription_thread.start()
    ASD_thread.start()    