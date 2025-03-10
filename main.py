import threading
import time
import queue
import traceback
import sys
import importlib
from Live_transcription.OnlineTranscription import RealtimeTranscriber
from audio_model.Classifier import AddressClassifier
from LLM import AI
import pyttsx3
# Import and then directly access the module to get up-to-date shared_state
import LIGHT_ASD.realtime3 as asd_module
from LIGHT_ASD.realtime3 import main as asd_main

import spacy

nlp = spacy.blank("en")  # Load English tokenizer

def remove_non_english_text(response):
    doc = nlp(response)
    return ' '.join([token.text for token in doc if token.is_alpha])  # Keep only alphabetic words

# Create thread-safe queues for communication between threads
transcription_queue = queue.Queue()
response_queue = queue.Queue()

# Initialize AI components
chat = AI()

def transcription_thread():
    """Handles real-time transcription and puts results in a queue"""
    with RealtimeTranscriber(prompt="This is a tech discussion, listen to english only") as transcriber:
        print("Speak now! (Press Ctrl+C to stop)")
        try:
            while True:
                result = transcriber.get_transcription()
                if result:
                    transcription_queue.put(result)
                time.sleep(0.05)  # Reduced sleep time for more frequent checks
        except KeyboardInterrupt:
            print("Stopping transcription...")

def addressing_thread():
    """Processes transcriptions and determines if AI should respond"""
    try:
        print("Initializing addressing classifier...")
        classifier = AddressClassifier()
        print("Addressing classifier initialized!")
        
        context = ""
        last_processed = time.time()
        process_interval = 0.2  # Process context every 200ms
        
        while True:
            # Get any new transcription
            try:
                while not transcription_queue.empty():
                    result = transcription_queue.get_nowait()
                    context += result + " "
                    print(f"Added to context: {result}")
            except queue.Empty:
                pass
            
            current_time = time.time()
            # Only process at intervals to avoid constant classification
            if current_time - last_processed > process_interval and context.strip():
                # Directly access the module's shared_state for up-to-date value
                current_shared_state = asd_module.shared_state
                print(f"Checking if addressing robot. Shared state: {current_shared_state}")
                
                if current_shared_state is None:
                    print("WARNING: shared_state is None, assuming False")
                    current_shared_state = False
                
                if current_shared_state:  # Check if speaker is looking at camera/active
                    try:
                        out = classifier.classify_text(context)
                        print(f"Classification result: {out}")
                        if out["is_addressing_robot"] and out['confidence'] > 0.6:
                            print(f"Context addressed to robot: {context}")
                            # Put in a separate thread to avoid blocking
                            threading.Thread(
                                target=generate_response, 
                                args=(context,)
                            ).start()
                            context = ""
                    except Exception as e:
                        print(f"Error in classification: {e}")
                        traceback.print_exc()
                
                # Reset context if it gets too long
                if len(context) >= 100:
                    print("Context too long, resetting")
                    context = ""
                    
                last_processed = current_time
            
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in addressing thread: {e}")
        traceback.print_exc()

def generate_response(prompt_text):
    """Generate AI response in a separate thread"""
    response = chat.generate_response(
        prompt=f"Input: {prompt_text}", 
        system_message="You are an AI model who gives to Live transcription message when you are addressed\n\ninput: I was reading about the use of AI in agriculture. What do you think,\noutput: <response>"
    )
    # print(f"AI: {response}")
    cleaned_response = remove_non_english_text(response)
    print(f"AI: {cleaned_response}")

    engine = pyttsx3.init()

    # Set speech rate to normal (default is around 200)
    engine.setProperty("rate", 140)  # Adjust this value if needed

    # Set voice to female
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)

    # Speak the cleaned text
    engine.say(cleaned_response)
    engine.runAndWait()
    response_queue.put(cleaned_response)

def asd_thread():
    """Runs the Active Speaker Detection in a separate thread"""
    try:
        asd_main(run_sub_audio_thread=True)
    except Exception as e:
        print(f"ASD thread error: {e}")

if __name__ == "__main__":
    # Create threads
    asd = threading.Thread(target=asd_thread)
    transcription = threading.Thread(target=transcription_thread)
    addressing = threading.Thread(target=addressing_thread)
    
    # Set as daemon threads so they exit when main program exits
    # asd.daemon = True
    # transcription.daemon = True
    # addressing.daemon = True
    
    # Start threads
    print("Starting all threads...")
    asd.start()
    time.sleep(1)  # Give ASD thread time to initialize
    transcription.start() 
    addressing.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down...")