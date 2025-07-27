# address_classifier_test.py
import threading
import time
import queue
from transformers import pipeline
import traceback

# Import your existing modules
from Live_transcription.Transcription3 import WhisperRealtimeTranscriber
# from Live_transcription.Transcription2 import RealTimeTranscriber
from LLM import AI

class AddressClassifierTest:
    def __init__(self):
        # Initialize the classification pipeline
        self.classifier = pipeline(
            "text-classification", 
            model=r"audio_model\distilbert-speaksense\checkpoint-1652", 
            tokenizer=r"audio_model\distilbert-speaksense\checkpoint-1652"
        )
        
        # Initialize transcriber and AI
        self.transcriber = WhisperRealtimeTranscriber()
        self.ai = AI()
        
        # Shared variables
        self.transcription_buffer = ""
        self.lock = threading.Lock()
        self.running = True
        
        print("Address Classifier Test initialized!")
        print("Starting transcription and classification...")
        
    def transcription_thread(self):
        """Handles real-time transcription"""
        try:
            # Start the transcriber
            self.transcriber.start_listening()
            self.transcriber.start_transcribing()
            
            while self.running:
                with self.lock:
                    if self.transcriber.last_transcription:
                        # Clean hallucinated words
                        hallucinated_words = ["See you next time", "!", "Thank you for watching", "thank you", "?"]
                        cleaned_transcription = self.transcriber.last_transcription
                        for word in hallucinated_words:
                            if word in cleaned_transcription:
                                cleaned_transcription = cleaned_transcription.replace(word, "")
                        
                        # Update buffer with last 1000 characters
                        self.transcription_buffer = cleaned_transcription[-1000:] if len(cleaned_transcription) > 1000 else cleaned_transcription
                        
                        print(f"Current transcription buffer ({len(self.transcription_buffer)} chars): {self.transcription_buffer}")
                
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Transcription thread error: {e}")
            traceback.print_exc()
        finally:
            self.transcriber.stop()
    
    def classification_thread(self):
        """Handles classification and LLM response"""
        last_processed_text = ""
        
        while self.running:
            try:
                with self.lock:
                    current_text = self.transcription_buffer
                
                # Only process if we have new text and it's not empty
                if current_text and current_text != last_processed_text and len(current_text.strip()) > 10:
                    print(f"\n--- CLASSIFYING TEXT ---")
                    print(f"Text to classify: {current_text}")
                    
                    # Run classification
                    result = self.classifier(current_text)
                    label = result[0]['label']
                    score = result[0]['score']
                    
                    print(f"Classification result: {label} (confidence: {score:.4f})")
                    
                    # Check if addressing robot (assuming LABEL_1 means addressing robot)
                    is_addressing = (label == 'LABEL_0' and score > 0.6)
                    
                    if is_addressing:
                        print(f"🤖 USER IS ADDRESSING THE ROBOT! Generating response...")
                        print(f"The context fed to it is:{current_text}")
                        # Generate AI response
                        # response = self.ai.generate_response(
                        #     prompt=f"Input: {current_text}",
                        #     system_message="You are an AI model who gives replies to conversation. Try your best to figure out what the user meant from their query, reply in 2 sentences"
                        # )
                        response = "__________________________________________________________________________________________________AI spoke_________________________________________________________________________________________________________"
                        
                        print(f"🤖 AI RESPONSE: {response}")
                        print("=" * 80)
                        
                        # Clear buffer after processing to avoid repeated responses
                        with self.lock:
                            self.transcriber.reset()
                    else:
                        print(f"❌ Not addressing robot (confidence too low or wrong label)")
                    
                    last_processed_text = current_text
                    print("-" * 50)
                
                time.sleep(0.2)  # Check every 200ms
                
            except Exception as e:
                print(f"Classification thread error: {e}")
                traceback.print_exc()
                time.sleep(1)  # Wait longer on error
    
    def run(self):
        """Start all threads and run the test"""
        try:
            # Create and start threads
            transcription_thread = threading.Thread(target=self.transcription_thread)
            classification_thread = threading.Thread(target=self.classification_thread)
            
            # Set as daemon threads
            transcription_thread.daemon = True
            classification_thread.daemon = True
            
            # Start threads
            transcription_thread.start()
            classification_thread.start()
            
            print("All threads started! Speak into your microphone...")
            print("Press Ctrl+C to stop")
            
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping Address Classifier Test...")
            self.running = False
        except Exception as e:
            print(f"Main thread error: {e}")
            traceback.print_exc()
        finally:
            self.running = False

if __name__ == "__main__":
    # Create and run the test
    test = AddressClassifierTest()
    test.run()