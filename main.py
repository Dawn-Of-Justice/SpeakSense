"""
SpeakSense: A multimodal deep learning system for detecting when a user is addressing
a virtual assistant by analyzing both audio and video in real-time.

This implementation features:
- Thread-safe state management
- Sliding window context system
- Improved turn-taking for natural conversation flow
- Producer-consumer pattern for component communication
- Robust error handling and recovery
"""

import threading
import time
import queue
import traceback
import os
import logging
from typing import Dict, List, Optional, Any

# Import your existing components
from Live_transcription.Transcription import WhisperRealtimeTranscriber
from audio_model.Classifier import AddressClassifier
from LLM import AI
import pyttsx3
# Import and then directly access the module to get up-to-date shared_state
import LIGHT_ASD.realtime3 as asd_module
from LIGHT_ASD.realtime3 import main as asd_main
import playsound
import spacy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("speaksense.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SpeakSense")

# Initialize spaCy
nlp = spacy.blank("en")  # Load English tokenizer

###################
# State Management
###################

class StateManager:
    """
    Centralized state management with thread safety.
    Controls the overall system state and acts as a communication hub.
    """
    
    def __init__(self):
        self.lock = threading.RLock()  # Reentrant lock for nested acquire calls
        self._ai_is_speaking = False
        self._system_active = True
        self._pause_transcription = False
        self._last_state_change = time.time()
        
        # Event listeners
        self.state_change_callbacks = []
    
    @property
    def ai_is_speaking(self) -> bool:
        """Thread-safe access to AI speaking state"""
        with self.lock:
            return self._ai_is_speaking
    
    @ai_is_speaking.setter
    def ai_is_speaking(self, value: bool):
        """Thread-safe update to AI speaking state with event notification"""
        with self.lock:
            old_value = self._ai_is_speaking
            self._ai_is_speaking = value
            self._last_state_change = time.time()
            
            # Notify listeners if state changed
            if old_value != value:
                for callback in self.state_change_callbacks:
                    try:
                        callback("ai_speaking", value)
                    except Exception as e:
                        logger.error(f"Error in state change callback: {e}")
    
    @property
    def system_active(self) -> bool:
        """Thread-safe access to system active state"""
        with self.lock:
            return self._system_active
    
    @system_active.setter
    def system_active(self, value: bool):
        """Thread-safe update to system active state"""
        with self.lock:
            self._system_active = value
            self._last_state_change = time.time()
    
    @property
    def pause_transcription(self) -> bool:
        """Thread-safe access to transcription pause state"""
        with self.lock:
            return self._pause_transcription
    
    @pause_transcription.setter
    def pause_transcription(self, value: bool):
        """Thread-safe update to transcription pause state"""
        with self.lock:
            self._pause_transcription = value
    
    def add_state_change_listener(self, callback):
        """Add a callback function to be called when state changes"""
        with self.lock:
            self.state_change_callbacks.append(callback)
    
    def remove_state_change_listener(self, callback):
        """Remove a callback function"""
        with self.lock:
            if callback in self.state_change_callbacks:
                self.state_change_callbacks.remove(callback)

class ContextManager:
    """
    Manages conversation context using a sliding window approach.
    Maintains the most recent utterances while preventing unbounded growth.
    """
    
    def __init__(self, max_tokens: int = 150):
        self.max_tokens = max_tokens
        self.context_tokens: List[str] = []
        self.lock = threading.Lock()
    
    def add(self, text: str):
        """Add new text to the context with sliding window"""
        if not text or text.isspace():
            return
            
        with self.lock:
            # Split and add new tokens
            new_tokens = text.split()
            self.context_tokens.extend(new_tokens)
            
            # Trim if needed
            if len(self.context_tokens) > self.max_tokens:
                self.context_tokens = self.context_tokens[-self.max_tokens:]
    
    def get_context(self) -> str:
        """Get the current context as a string"""
        with self.lock:
            return " ".join(self.context_tokens)
    
    def clear(self):
        """Clear the entire context"""
        with self.lock:
            self.context_tokens = []

class ConversationManager:
    """
    Manages the turn-taking in conversation.
    Handles transitions between user speaking and AI responding states.
    """
    
    # Define possible conversation states
    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    AI_RESPONDING = "ai_responding"
    
    def __init__(self, state_manager: StateManager):
        self.state = self.IDLE
        self.last_user_activity = 0
        self.user_silence_threshold = 1.5  # seconds
        self.lock = threading.Lock()
        self.state_manager = state_manager
        
        # Register for state changes
        self.state_manager.add_state_change_listener(self._handle_state_change)
    
    def _handle_state_change(self, state_name: str, value: Any):
        """Handle state changes from the StateManager"""
        if state_name == "ai_speaking" and value is True:
            with self.lock:
                self.state = self.AI_RESPONDING
        elif state_name == "ai_speaking" and value is False:
            with self.lock:
                self.state = self.IDLE
    
    def user_activity_detected(self) -> bool:
        """
        Record user activity and update state if needed
        Returns True if state changed
        """
        with self.lock:
            self.last_user_activity = time.time()
            
            # Only change state if we're idle and AI is not speaking
            if self.state == self.IDLE and not self.state_manager.ai_is_speaking:
                self.state = self.USER_SPEAKING
                return True  # State changed
            
            return False
    
    def check_user_silence(self) -> bool:
        """
        Check if user has been silent long enough to consider their turn complete
        Returns True if user finished speaking
        """
        with self.lock:
            if self.state == self.USER_SPEAKING:
                if time.time() - self.last_user_activity > self.user_silence_threshold:
                    logger.info("User silence detected - turn complete")
                    self.state = self.IDLE
                    return True  # User finished speaking
            
            return False
    
    def get_state(self) -> str:
        """Get the current conversation state"""
        with self.lock:
            return self.state

###################
# Processing Components
###################

class TranscriptionProcessor:
    """
    Processes raw transcriptions to clean and normalize text
    """
    
    def __init__(self):
        # List of words that tend to be hallucinated by the transcription model
        self.hallucinated_words = [
            "See you next time", "!", "Thank you for watching", 
            "thank you", "?", "goodbye", "thanks for watching"
        ]
    
    def process(self, text: str) -> str:
        """
        Process a transcription result to:
        1. Remove hallucinated words
        2. Normalize text (remove extra spaces, etc.)
        3. Filter non-English text
        """
        if not text:
            return ""
            
        # Remove hallucinated words
        processed = text
        for word in self.hallucinated_words:
            processed = processed.replace(word, "")
        
        # Normalize spaces
        processed = " ".join(processed.split())
        
        # Filter to keep only English words
        doc = nlp(processed)
        filtered = ' '.join([token.text for token in doc if token.is_alpha or token.text.isspace()])
        
        return filtered

class ResponseGenerator:
    """
    Generates AI responses based on conversation context
    """
    
    def __init__(self, state_manager: StateManager):
        self.ai = AI()
        self.state_manager = state_manager
        self.system_message = (
            "You are an AI model who gives replies to crippled conversation "
            "try your best to figure out what the user meant from the cut off "
            "or maybe not so cut of user query, reply in 2 sentences"
        )
    
    def generate(self, context: str) -> str:
        """Generate an AI response to the given context"""
        try:
            # Inform the system that AI is speaking
            self.state_manager.ai_is_speaking = True
            
            # Generate the response
            response = self.ai.generate_response(
                prompt=f"Input: {context}",
                system_message=self.system_message
            )
            
            # Clean up response
            cleaned_response = self._clean_response(response)
            logger.info(f"Generated response: {cleaned_response}")
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I couldn't understand that properly. Could you please repeat?"
        finally:
            # Always reset speaking state, even if there was an error
            self.state_manager.ai_is_speaking = False
    
    def _clean_response(self, response: str) -> str:
        """Clean and filter the generated response"""
        doc = nlp(response)
        return ' '.join([token.text for token in doc if token.is_alpha or token.text.isspace() or token.text in ",.!?"])

class SpeechSynthesizer:
    """
    Converts text to speech and plays it
    """
    
    def __init__(self, voice_id: int = 1, rate: int = 140):
        self.voice_id = voice_id
        self.rate = rate
        self.temp_file = "temp_speech.wav"
    
    def speak(self, text: str):
        """Convert text to speech and play it"""
        try:
            # Initialize the TTS engine
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            
            # Set voice
            voices = engine.getProperty("voices")
            if len(voices) > self.voice_id:
                engine.setProperty("voice", voices[self.voice_id].id)
            
            # Save to file and play
            engine.save_to_file(text, self.temp_file)
            engine.runAndWait()
            
            # Play the audio
            playsound.playsound(self.temp_file)
            
            # Clean up the file
            if os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                except:
                    pass
                
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")

###################
# Thread Workers
###################

class TranscriptionWorker:
    """
    Worker thread for real-time transcription
    """
    
    def __init__(
        self, 
        state_manager: StateManager, 
        output_queue: queue.Queue,
        processor: TranscriptionProcessor
    ):
        self.state_manager = state_manager
        self.output_queue = output_queue
        self.processor = processor
        self.transcriber = WhisperRealtimeTranscriber()
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the transcription worker thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Transcription worker already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the transcription worker thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.transcriber.stop()
    
    def _run(self):
        """Main worker thread function"""
        try:
            # Initialize the transcriber
            self.transcriber.start_listening()
            self.transcriber.start_transcribing()
            
            logger.info("Transcription worker started")
            
            last_transcription = ""
            
            # Main processing loop
            while self.running and self.state_manager.system_active:
                # Only process transcription when AI is not speaking
                if not self.state_manager.ai_is_speaking and not self.state_manager.pause_transcription:
                    current_transcription = self.transcriber.last_transcription
                    
                    # Only process if we have new transcription
                    if current_transcription and current_transcription != last_transcription:
                        # Process the transcription
                        processed_text = self.processor.process(current_transcription)
                        
                        if processed_text:
                            # Put the processed text in the output queue
                            self.output_queue.put(processed_text)
                            logger.debug(f"Transcribed: {processed_text}")
                        
                        last_transcription = current_transcription
                
                # Sleep to avoid tight loops
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in transcription worker: {e}")
            traceback.print_exc()
        finally:
            # Ensure resources are properly cleaned up
            try:
                self.transcriber.stop()
            except:
                pass
            logger.info("Transcription worker stopped")

class AddressingWorker:
    """
    Worker thread for determining if the user is addressing the assistant
    """
    
    def __init__(
        self, 
        state_manager: StateManager,
        conversation_manager: ConversationManager,
        context_manager: ContextManager,
        transcription_queue: queue.Queue,
        addressing_queue: queue.Queue
    ):
        self.state_manager = state_manager
        self.conversation_manager = conversation_manager
        self.context_manager = context_manager
        self.transcription_queue = transcription_queue
        self.addressing_queue = addressing_queue
        self.classifier = None
        self.running = False
        self.thread = None
        self.process_interval = 0.3  # seconds
        self.last_processed = 0
        self.confidence_threshold = 0.6
    
    def start(self):
        """Start the addressing worker thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Addressing worker already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the addressing worker thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
    
    def _run(self):
        """Main worker thread function"""
        try:
            # Initialize the classifier
            logger.info("Initializing addressing classifier...")
            self.classifier = AddressClassifier()
            logger.info("Addressing classifier initialized")
            
            # Main processing loop
            while self.running and self.state_manager.system_active:
                # Process any new transcriptions
                try:
                    while not self.transcription_queue.empty():
                        text = self.transcription_queue.get_nowait()
                        self.context_manager.add(text)
                        
                        # Mark activity in the conversation manager
                        self.conversation_manager.user_activity_detected()
                except queue.Empty:
                    pass
                
                # Check if user has finished speaking
                user_turn_complete = self.conversation_manager.check_user_silence()
                
                # Only process at intervals and if conversation state is appropriate
                current_time = time.time()
                if (current_time - self.last_processed > self.process_interval and 
                    user_turn_complete and 
                    self.conversation_manager.get_state() == ConversationManager.IDLE):
                    
                    context = self.context_manager.get_context()
                    
                    # Only process if we have context and the ASD system thinks user is looking at camera
                    current_shared_state = asd_module.shared_state
                    if current_shared_state is None:
                        logger.warning("ASD shared_state is None, assuming False")
                        current_shared_state = False
                    
                    if context and current_shared_state:
                        try:
                            # Classify if the user is addressing the assistant
                            classification = self.classifier.classify_text(context)
                            logger.debug(f"Classification result: {classification}")
                            
                            # Check if classification meets our threshold
                            if (classification.get("is_addressing_robot", False) and 
                                classification.get("confidence", 0) > self.confidence_threshold):
                                
                                logger.info(f"User is addressing assistant: {context}")
                                
                                # Queue for processing
                                self.addressing_queue.put(context)
                                
                                # Clear context after processing
                                self.context_manager.clear()
                            
                        except Exception as e:
                            logger.error(f"Error in addressing classification: {e}")
                    
                    self.last_processed = current_time
                
                # Sleep to avoid tight loops
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in addressing worker: {e}")
            traceback.print_exc()

class ResponseWorker:
    """
    Worker thread for generating and speaking AI responses
    """
    
    def __init__(
        self, 
        state_manager: StateManager,
        addressing_queue: queue.Queue,
        response_generator: ResponseGenerator,
        speech_synthesizer: SpeechSynthesizer
    ):
        self.state_manager = state_manager
        self.addressing_queue = addressing_queue
        self.response_generator = response_generator
        self.speech_synthesizer = speech_synthesizer
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the response worker thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Response worker already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the response worker thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
    
    def _run(self):
        """Main worker thread function"""
        try:
            logger.info("Response worker started")
            
            # Main processing loop
            while self.running and self.state_manager.system_active:
                try:
                    # Wait for an addressing event with a timeout
                    try:
                        context = self.addressing_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    
                    # Generate a response if we got a context
                    if context:
                        # Temporarily pause transcription while generating and speaking
                        self.state_manager.pause_transcription = True
                        
                        # Generate the response
                        response = self.response_generator.generate(context)
                        
                        # Speak the response
                        if response:
                            self.speech_synthesizer.speak(response)
                        
                        # Resume transcription
                        self.state_manager.pause_transcription = False
                        
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                
                # Small sleep to avoid tight loops
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in response worker: {e}")
            traceback.print_exc()
        finally:
            logger.info("Response worker stopped")

class ASDWorker:
    """
    Worker thread for Active Speaker Detection
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the ASD worker thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("ASD worker already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the ASD worker thread"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
    
    def _run(self):
        """Main worker thread function"""
        try:
            logger.info("ASD worker starting...")
            
            # Run the ASD main function
            asd_main(run_sub_audio_thread=True)
            
        except Exception as e:
            logger.error(f"Error in ASD worker: {e}")
            traceback.print_exc()
        finally:
            logger.info("ASD worker stopped")

###################
# Main Application
###################

class SpeakSense:
    """
    Main application class that coordinates all components
    """
    
    def __init__(self):
        # Create queues for inter-thread communication
        self.transcription_queue = queue.Queue()
        self.addressing_queue = queue.Queue()
        
        # Create core components
        self.state_manager = StateManager()
        self.context_manager = ContextManager()
        self.conversation_manager = ConversationManager(self.state_manager)
        
        # Create processing components
        self.transcription_processor = TranscriptionProcessor()
        self.response_generator = ResponseGenerator(self.state_manager)
        self.speech_synthesizer = SpeechSynthesizer()
        
        # Create worker threads
        self.transcription_worker = TranscriptionWorker(
            self.state_manager,
            self.transcription_queue,
            self.transcription_processor
        )
        
        self.addressing_worker = AddressingWorker(
            self.state_manager,
            self.conversation_manager,
            self.context_manager,
            self.transcription_queue,
            self.addressing_queue
        )
        
        self.response_worker = ResponseWorker(
            self.state_manager,
            self.addressing_queue,
            self.response_generator,
            self.speech_synthesizer
        )
        
        self.asd_worker = ASDWorker(self.state_manager)
    
    def start(self):
        """Start all components of the system"""
        try:
            logger.info("Starting SpeakSense system...")
            
            # Start ASD first and give it time to initialize
            self.asd_worker.start()
            time.sleep(1)
            
            # Start other components
            self.transcription_worker.start()
            self.addressing_worker.start()
            self.response_worker.start()
            
            logger.info("SpeakSense system started")
            
            # Keep the main thread alive
            try:
                while self.state_manager.system_active:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                self.stop()
            
        except Exception as e:
            logger.error(f"Error starting SpeakSense: {e}")
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """Stop all components of the system"""
        logger.info("Stopping SpeakSense system...")
        
        # Set system to inactive
        self.state_manager.system_active = False
        
        # Stop all workers
        self.response_worker.stop()
        self.addressing_worker.stop()
        self.transcription_worker.stop()
        self.asd_worker.stop()
        
        logger.info("SpeakSense system stopped")

if __name__ == "__main__":
    # Create and start the SpeakSense system
    system = SpeakSense()
    system.start()