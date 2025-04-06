import threading
import time
import queue
import traceback
import sys
import importlib
# from Live_transcription.OnlineTranscription import RealtimeTranscriber
# from Live_transcription.Transcription import WhisperRealtimeTranscriber
from Live_transcription.Transcription3 import WhisperRealtimeTranscriber
from audio_model.Classifier import AddressClassifier
from LLM import AI
import pyttsx3
# Import and then directly access the module to get up-to-date shared_state
import LIGHT_ASD.realtime3 as asd_module
from LIGHT_ASD.realtime3 import main as asd_main
import playsound
import spacy

nlp = spacy.blank("en")  # Load English tokenizer

def remove_non_english_text(response):
    doc = nlp(response)
    return ' '.join([token.text for token in doc if token.is_alpha])  # Keep only alphabetic words

# Create thread-safe queues for communication between threads
transcription_queue = queue.Queue()
response_queue = queue.Queue()
clear_state = False
transcribed_stuff = None
ai_is_speaking = False

# Initialize AI components
chat = AI()

lock = threading.Condition() 
            
def transcription_thread():
    """Handles real-time transcription and puts results in a queue"""
    global clear_state, transcribed_stuff, ai_is_speaking
    
    transcriber = WhisperRealtimeTranscriber()
    
    try:
        # Start listening and transcribing
        transcriber.start_listening()
        transcriber.start_transcribing()
        
        # Keep running until Ctrl+C
        print("Press Ctrl+C to stop transcription")
        while True:
            with lock:
                # Only process transcription when AI is not speaking
                if not ai_is_speaking:
                    if clear_state:
                        transcriber.last_transcription = ""
                        clear_state = False
                        print(f"Cleared Context: {transcriber.last_transcription}")
                    if transcriber.last_transcription:
                        hallucinated_words = ["See you next time", "!", "Thank you for watching", "thank you", "?"]
                        for word in hallucinated_words:
                            if word in transcriber.last_transcription:
                                transcriber.last_transcription = transcriber.last_transcription.replace(word, "")
                        transcribed_stuff = transcriber.last_transcription
                time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        transcriber.stop()

def addressing_thread():
    """Processes transcriptions and determines if AI should respond"""
    global clear_state, transcribed_stuff
    try:
        print("Initializing addressing classifier...")
        classifier = AddressClassifier()
        print("Addressing classifier initialized!")
        
        context = ""
        last_processed = time.time()
        process_interval = 0.2  # Process context every 200ms
        
        while True:
            # Get any new transcription
            # try:
            #     while not transcription_queue.empty():
            #         result = transcription_queue.get_nowait()
            #         context += result + " "
            #         # print(f"Added to context: {result}")
            # except queue.Empty:
            #     pass
            
            if transcribed_stuff:
                print(f"Transcribed_stuff: {transcribed_stuff}")
                context = transcribed_stuff
            
            current_time = time.time()
            # Only process at intervals to avoid constant classification
            if current_time - last_processed > process_interval and context.strip():
                # Directly access the module's shared_state for up-to-date value
                current_shared_state = asd_module.shared_state
                # print(f"Checking if addressing robot. Shared state: {current_shared_state}")
                
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
                            # threading.Thread(
                            #     target=generate_response, 
                            #     args=(context,)
                            # ).start()
                            print("AI is speaking")
                            with lock:
                                generate_response(context)
                            print("AI is done speaking")
                            # context = ""
                            # transcription_queue.queue.clear()
                            # clear_state = True
                    except Exception as e:
                        print(f"Error in classification: {e}")
                        traceback.print_exc()
                
                # Reset context if it gets too long
                if len(context) >= 5000:
                    print("Context too long, resetting")
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
        
        response = chat.generate_response(
            prompt=f"Input: {prompt_text}", 
            system_message="You are an AI model who gives replies to crippled conversation try your best to figure out what the user meant from the cut off or maybe not so cut of user query, reply in 2 sentences"
        )
        cleaned_response = remove_non_english_text(response)
        print(f"AI: {cleaned_response}")

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
    transcription.daemon = True
    addressing.daemon = True
    
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