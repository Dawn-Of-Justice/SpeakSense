"""
SpeakSense: A multimodal deep learning system for detecting when a user is addressing
a virtual assistant by analyzing both audio and video in real-time.

This implementation includes enhanced audio feedback prevention to stop the system
from listening to its own TTS output.
"""

import threading
import time
import queue
import traceback
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Any

import pyaudio

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
        
        # Audio feedback prevention
        self._audio_suppression_active = False
        self._last_human_energy = 0.02  # Starting baseline for human speech energy
        
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
            
            # When AI starts speaking, activate audio suppression
            if value and not old_value:
                self._audio_suppression_active = True
                logger.debug("Audio suppression activated")
            
            # When AI stops speaking, keep suppression active for a short delay
            if not value and old_value:
                # Audio suppression will be deactivated after a cooldown period
                threading.Timer(0.5, self._deactivate_suppression).start()
            
            # Notify listeners if state changed
            if old_value != value:
                for callback in self.state_change_callbacks:
                    try:
                        callback("ai_speaking", value)
                    except Exception as e:
                        logger.error(f"Error in state change callback: {e}")
    
    def _deactivate_suppression(self):
        """Deactivate audio suppression after cooldown period"""
        with self.lock:
            self._audio_suppression_active = False
            logger.debug("Audio suppression deactivated")
    
    @property
    def audio_suppression_active(self) -> bool:
        """Thread-safe access to audio suppression state"""
        with self.lock:
            return self._audio_suppression_active
    
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
    
    @property
    def last_human_energy(self) -> float:
        """Get the last recorded human speech energy level"""
        with self.lock:
            return self._last_human_energy
    
    @last_human_energy.setter
    def last_human_energy(self, value: float):
        """Update the last recorded human speech energy level"""
        with self.lock:
            # Only update when not speaking to avoid contamination with AI speech
            if not self._ai_is_speaking and not self._audio_suppression_active:
                self._last_human_energy = value
    
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
# Audio Analysis
###################

class AudioAnalyzer:
    """
    Analyzes audio for voice activity detection and feedback prevention
    """
    
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.energy_threshold = 0.02  # Base threshold for voice activity
        self.energy_adjustment_rate = 0.1  # How quickly to adapt the threshold
        
        # Keep a history of recent audio to detect patterns
        self.recent_energies = []
        self.max_history = 20
    
    def is_human_voice(self, audio_np: np.ndarray) -> bool:
        """
        Determine if audio is likely from a human (vs. from speakers)
        
        Returns:
            bool: True if likely human speech
        """
        if len(audio_np) == 0:
            return False
            
        # Calculate energy metrics
        energy = np.abs(audio_np).mean()
        energy_variance = np.var(np.abs(audio_np))
        
        # Update energy history
        self.recent_energies.append(energy)
        if len(self.recent_energies) > self.max_history:
            self.recent_energies.pop(0)
        
        # Get baseline from state manager
        baseline_energy = self.state_manager.last_human_energy
        
        # If AI is speaking, use stricter detection to avoid feedback
        if self.state_manager.ai_is_speaking or self.state_manager.audio_suppression_active:
            # During AI speech, require significantly higher energy and variance
            # to consider it human speech (to avoid feedback)
            energy_ratio = energy / baseline_energy if baseline_energy > 0 else 1.0
            
            # Speaker output usually has lower variance than human speech
            is_human = (
                energy_variance > 0.0008 and  # Higher variance threshold
                energy_ratio > 1.8 and        # Much louder than baseline
                energy > self.energy_threshold * 1.5  # Higher absolute threshold
            )
            
            if is_human:
                logger.debug(f"Detected human voice during AI speech: energy={energy:.4f}, variance={energy_variance:.6f}, ratio={energy_ratio:.2f}")
            
            return is_human
        
        # Normal operation (AI not speaking)
        is_human = energy > self.energy_threshold
        
        # Update baseline energy for future comparisons
        if is_human:
            # Smooth update to the energy baseline
            new_baseline = baseline_energy * (1 - self.energy_adjustment_rate) + energy * self.energy_adjustment_rate
            self.state_manager.last_human_energy = new_baseline
        
        return is_human


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
            # State manager will handle the audio suppression cooldown
            self.state_manager.ai_is_speaking = False
    
    def _clean_response(self, response: str) -> str:
        """Clean and filter the generated response"""
        doc = nlp(response)
        return ' '.join([token.text for token in doc if token.is_alpha or token.text.isspace() or token.text in ",.!?"])


class SpeechSynthesizer:
    """
    Converts text to speech and plays it
    """
    
    def __init__(self, voice_id: int = 1, rate: int = 140, transcription_worker=None):
        self.voice_id = voice_id
        self.rate = rate
        self.temp_file = "temp_speech.wav"
        self.transcription_worker = transcription_worker 
    
    def speak(self, text: str):
        """
        Convert text to speech and play it with complete audio isolation
        to prevent the system from hearing itself.
        """
        try:
            # 1. Store reference to the transcription worker's audio stream
            transcriber = self.transcription_worker.transcriber
            
            # 2. COMPLETELY STOP the audio stream before TTS
            if hasattr(transcriber, 'stream') and transcriber.stream and transcriber.stream.is_active():
                logger.info("Stopping audio stream for TTS playback")
                transcriber.stream.stop_stream()
            
            # 3. Clear all buffers and queues
            with transcriber.transcription_lock:
                transcriber.audio_buffer = np.array([], dtype=np.float32)
                # Clear the audio queue
                while not transcriber.audio_queue.empty():
                    try:
                        transcriber.audio_queue.get_nowait()
                    except queue.Empty:
                        break
            
            # 4. Initialize the TTS engine
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            
            # Set voice
            voices = engine.getProperty("voices")
            if len(voices) > self.voice_id:
                engine.setProperty("voice", voices[self.voice_id].id)
            
            # 5. Generate and save TTS audio
            logger.info(f"Generating TTS for: {text}")
            engine.save_to_file(text, self.temp_file)
            engine.runAndWait()
            
            # 6. Play the audio while microphone is stopped
            logger.info("Playing TTS audio with microphone muted")
            playsound.playsound(self.temp_file)
            
            # 7. Add a significant delay before resuming audio capture
            time.sleep(0.5)
            
            # 8. RESTART the audio stream
            if hasattr(transcriber, 'stream'):
                logger.info("Restarting audio stream after TTS")
                if not transcriber.stream.is_active():
                    transcriber.stream.start_stream()
            
            # Clean up the file
            if os.path.exists(self.temp_file):
                try:
                    os.remove(self.temp_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            logger.error(traceback.format_exc())
            
            # Ensure the audio stream is restarted even on error
            try:
                transcriber = self.transcription_worker.transcriber
                if hasattr(transcriber, 'stream') and transcriber.stream and not transcriber.stream.is_active():
                    transcriber.stream.start_stream()
            except:
                pass


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
        processor: TranscriptionProcessor,
        audio_analyzer: AudioAnalyzer
    ):
        self.state_manager = state_manager
        self.output_queue = output_queue
        self.processor = processor
        self.audio_analyzer = audio_analyzer
        self.transcriber = WhisperRealtimeTranscriber(optimize=True)
        self.running = False
        self.thread = None
        
        # Audio buffer for analysis
        self.current_audio_buffer = np.array([], dtype=np.float32)
    
    def start(self):
        """Start the transcription worker thread"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning("Transcription worker already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def restart_audio_stream(self):
        """Restart the audio stream after it has been stopped"""
        if hasattr(self.transcriber, 'stream') and self.transcriber.stream:
            if not self.transcriber.stream.is_active():
                try:
                    self.transcriber.stream.start_stream()
                    logger.info("Audio stream restarted")
                except Exception as e:
                    logger.error(f"Error restarting audio stream: {e}")
    
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
                try:
                    # CRITICAL: Skip all audio processing when AI is speaking or
                    # audio suppression is active to prevent feedback
                    if self.state_manager.ai_is_speaking or self.state_manager.audio_suppression_active:
                        # Clear audio buffers to prevent processing AI's own speech
                        with self.transcriber.transcription_lock:
                            self.transcriber.audio_buffer = np.array([], dtype=np.float32)
                            
                            # Also clear any queued audio chunks
                            while not self.transcriber.audio_queue.empty():
                                try:
                                    self.transcriber.audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                        
                        time.sleep(0.1)
                        continue
                    
                    # Get the current audio for analysis (if accessible in your implementation)
                    if hasattr(self.transcriber, 'audio_buffer'):
                        with self.transcriber.transcription_lock:
                            # Make a copy to avoid thread issues
                            current_audio = np.copy(self.transcriber.audio_buffer) 
                        
                        # Check if audio contains human voice
                        if len(current_audio) > 0 and self.audio_analyzer.is_human_voice(current_audio):
                            # Process transcription only for human voice
                            current_transcription = self.transcriber.get_transcription()["text"]
                            
                            # Only process if we have new transcription with decent confidence
                            if (current_transcription and 
                                current_transcription != last_transcription and
                                self.transcriber.get_transcription()["confidence"] > 0.4):
                                
                                # Process the transcription
                                processed_text = self.processor.process(current_transcription)
                                
                                if processed_text:
                                    # Put the processed text in the output queue
                                    self.output_queue.put(processed_text)
                                    logger.debug(f"Transcribed: {processed_text}")
                                
                                last_transcription = current_transcription
                except Exception as e:
                    logger.error(f"Error in transcription processing: {e}")
                
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
                # Skip processing during audio suppression
                if self.state_manager.ai_is_speaking or self.state_manager.audio_suppression_active:
                    time.sleep(0.1)
                    continue
                
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
        speech_synthesizer: SpeechSynthesizer,
        transcription_worker: 'TranscriptionWorker'  # For direct buffer access
    ):
        self.state_manager = state_manager
        self.addressing_queue = addressing_queue
        self.response_generator = response_generator
        self.speech_synthesizer = speech_synthesizer
        self.transcription_worker = transcription_worker
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
                        # Set flags to prevent audio processing
                        self.state_manager.pause_transcription = True
                        self.state_manager.ai_is_speaking = True
                        
                        # Small delay to ensure state changes propagate
                        time.sleep(0.1)
                        
                        # Generate the response
                        response = self.response_generator.generate(context)
                        
                        # Speak the response - this will now handle stopping and restarting the audio stream
                        if response:
                            self.speech_synthesizer.speak(response)
                        
                        # Resume normal state - the speak method now handles restarting the audio stream
                        self.state_manager.ai_is_speaking = False
                        self.state_manager.pause_transcription = False
                        
                        # Ensure we're back to clean state before processing next input
                        time.sleep(1.0)
                        
                except Exception as e:
                    logger.error(f"Error processing response: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Ensure audio stream is restarted on error
                    self.state_manager.ai_is_speaking = False
                    self.state_manager.pause_transcription = False
                    
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
        self.audio_analyzer = AudioAnalyzer(self.state_manager)
        
        # Create processing components
        self.transcription_processor = TranscriptionProcessor()
        self.response_generator = ResponseGenerator(self.state_manager)
        
        # Create worker threads (note the circular dependency handled in a clean way)
        self.transcription_worker = TranscriptionWorker(
            self.state_manager,
            self.transcription_queue,
            self.transcription_processor,
            self.audio_analyzer
        )

        self.speech_synthesizer = SpeechSynthesizer(
            transcription_worker=self.transcription_worker
        )
        
        self.addressing_worker = AddressingWorker(
            self.state_manager,
            self.conversation_manager,
            self.context_manager,
            self.transcription_queue,
            self.addressing_queue
        )
        
        # Create response worker with transcription_worker reference for buffer access
        self.response_worker = ResponseWorker(
            self.state_manager,
            self.addressing_queue,
            self.response_generator,
            self.speech_synthesizer,
            self.transcription_worker  # Pass reference for direct buffer clearing
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

###################
# Modified WhisperRealtimeTranscriber
###################

class WhisperRealtimeTranscriber:
    """
    Real-time audio transcription using OpenAI's Whisper model.
    Features:
    - Dynamic silence detection
    - Efficient buffering
    - Smart segmentation
    - Non-blocking transcription
    - Confidence scoring
    - Resource management
    """
    
    def __init__(
        self, 
        model_name="openai/whisper-tiny", 
        device=None, 
        optimize=True,
        sample_rate=16000
    ):
        """
        Initialize the transcriber with the specified model and settings
        
        Args:
            model_name (str): Whisper model name/path
            device (str): Device to use ('cuda', 'cpu', etc.)
            optimize (bool): Whether to optimize model for inference
            sample_rate (int): Audio sample rate in Hz
        """
        # Import necessary libraries here to avoid circular imports
        import torch
        import numpy as np
        import pyaudio
        import threading
        import queue
        import time
        import os
        import logging
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        # Set up logging
        self.logger = logging.getLogger("WhisperTranscriber")
        
        # Set device to CUDA if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load Whisper model and processor
        self.logger.info(f"Loading Whisper model: {model_name}")
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
            
            # Set model to English-only for better performance
            self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="en",  # Explicitly specify English
                task="transcribe"
            )
            
            # Optimize model if requested and on GPU
            if optimize and self.device == "cuda":
                self.model = self.model.half()  # Use FP16 for faster inference
                self.logger.info("Using half precision (FP16) for faster inference")
                
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
        # Audio configurations
        self.sample_rate = sample_rate
        self.chunk_size = 4000
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        try:
            self.audio_interface = pyaudio.PyAudio()
        except Exception as e:
            self.logger.error(f"Error initializing PyAudio: {e}")
            raise
        
        # Create a queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Tracking for continuous transcription
        self.is_running = False
        self.stream = None
        self.processing_thread = None
        self.transcription_lock = threading.Lock()
        
        # Audio buffer and state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_max_size = self.sample_rate * 30  # 30 seconds max
        self.last_transcription = ""
        self.last_confidence = 0.0
        self.last_timestamps = None
        
        # Silence detection settings
        self.base_silence_threshold = 0.01
        self.silence_threshold = self.base_silence_threshold
        self.silence_counter = 0
        self.max_silence_chunks = 8  # About 2 seconds of silence
        
        # Dynamic threshold calibration
        self.calibration_samples = []
        self.calibration_period = 100  # Calibrate every 100 chunks
        self.calibration_counter = 0
        
        # Performance metrics
        self.transcription_count = 0
        self.total_processing_time = 0
        self.total_audio_duration = 0
    
    def start_listening(self):
        """Start capturing audio from microphone"""
        if self.is_running:
            self.logger.warning("Already listening")
            return
            
        try:
            self.stream = self.audio_interface.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_running = True
            self.stream.start_stream()
            self.logger.info("Audio capture started - listening to microphone")
        except Exception as e:
            self.logger.error(f"Error starting audio stream: {e}")
            self.stop()
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            self.logger.debug(f"Audio callback status: {status}")
            
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _calibrate_silence_threshold(self, audio_level):
        """Dynamically adjust silence threshold based on ambient noise"""
        self.calibration_samples.append(audio_level)
        self.calibration_counter += 1
        
        # Recalibrate periodically
        if self.calibration_counter >= self.calibration_period:
            if len(self.calibration_samples) > 10:  # Need enough samples
                # Use a low percentile as baseline noise level
                noise_level = np.percentile(self.calibration_samples, 10)
                # Set threshold to be slightly above the noise level
                self.silence_threshold = max(self.base_silence_threshold, noise_level * 1.5)
                self.logger.debug(f"Recalibrated silence threshold to {self.silence_threshold:.5f}")
                
            # Reset calibration state
            self.calibration_samples = self.calibration_samples[-10:]  # Keep some history
            self.calibration_counter = 0
    
    def _process_audio(self):
        """Process audio chunks from the queue and manage transcription"""
        self.logger.info("Audio processing thread started")
        last_transcription_time = time.time()
        
        while self.is_running:
            try:
                # Process multiple chunks at once to reduce overhead
                chunks = []
                try:
                    # Wait for at least one chunk but don't block too long
                    first_chunk = self.audio_queue.get(timeout=0.5)
                    chunks.append(first_chunk)
                    
                    # Get any additional chunks that are ready (up to 10)
                    for _ in range(9):  # Max 10 chunks total
                        if self.audio_queue.empty():
                            break
                        chunks.append(self.audio_queue.get_nowait())
                except queue.Empty:
                    # No chunks available, wait and try again
                    time.sleep(0.1)
                    continue
                
                if chunks:
                    # Convert all chunks at once for efficiency
                    audio_data = b''.join(chunks)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Calculate audio level for this segment
                    audio_level = np.abs(audio_np).mean()
                    
                    # Calibrate silence threshold
                    self._calibrate_silence_threshold(audio_level)
                    
                    # Check for silence
                    is_silent = audio_level < self.silence_threshold
                    
                    if is_silent:
                        self.silence_counter += 1
                        self.logger.debug(f"Silence detected ({self.silence_counter}/{self.max_silence_chunks})")
                    else:
                        if self.silence_counter > 0:
                            self.logger.debug("Audio activity resumed")
                        self.silence_counter = 0
                    
                    # Add to buffer
                    self.audio_buffer = np.append(self.audio_buffer, audio_np)
                    
                    # Enforce maximum buffer size (circular buffer approach)
                    if len(self.audio_buffer) > self.buffer_max_size:
                        # Keep most recent audio up to buffer_max_size
                        self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]
                    
                    # Check if we should transcribe
                    current_time = time.time()
                    buffer_duration = len(self.audio_buffer) / self.sample_rate
                    time_since_last = current_time - last_transcription_time
                    
                    # Transcription triggers:
                    # 1. Significant silence after speech (end of utterance)
                    # 2. Buffer getting large (ongoing speech)
                    # 3. Minimum time since last transcription (avoid too frequent)
                    significant_silence = (
                        self.silence_counter >= self.max_silence_chunks and 
                        buffer_duration > 1.0
                    )
                    buffer_full = buffer_duration >= 5.0
                    time_threshold = time_since_last >= 3.0 and buffer_duration >= 1.0
                    
                    if (significant_silence or buffer_full or time_threshold) and buffer_duration > 0.5:
                        # Launch transcription in a separate thread to avoid blocking
                        threading.Thread(
                            target=self._transcribe_buffer,
                            daemon=True
                        ).start()
                        
                        last_transcription_time = current_time
            
            except Exception as e:
                self.logger.error(f"Error in audio processing: {e}")
                time.sleep(0.1)  # Avoid tight loop on error
    
    def _transcribe_buffer(self):
        """Transcribe the current audio buffer"""
        # Import torch dynamically
        import torch
        import traceback
        import numpy as np
        
        # Make a copy of the buffer to allow concurrent processing
        with self.transcription_lock:
            audio_to_process = np.copy(self.audio_buffer)
            
            # If significant silence, keep a small portion for context in next transcription
            if self.silence_counter >= self.max_silence_chunks:
                # Keep a small portion for context
                self.audio_buffer = self.audio_buffer[-int(self.sample_rate * 0.5):]
                self.silence_counter = 0
            else:
                # For continuous speech, keep more context
                self.audio_buffer = self.audio_buffer[-int(self.sample_rate * 2):]
        
        # Skip if buffer is too small
        if len(audio_to_process) < self.sample_rate * 0.5:  # At least 0.5 seconds
            self.logger.debug("Buffer too small, skipping transcription")
            return
        
        try:
            # Measure transcription performance
            start_time = time.time()
            buffer_duration = len(audio_to_process) / self.sample_rate
            
            # Process audio for Whisper input
            processed_features = self.processor(
                audio_to_process, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            
            # Get input features and move to device - handle both possible return types
            if hasattr(processed_features, "input_features"):
                input_features = processed_features.input_features
            elif isinstance(processed_features, dict) and "input_features" in processed_features:
                input_features = processed_features["input_features"]
            else:
                self.logger.error(f"Unexpected processor output format: {type(processed_features)}")
                self.logger.debug(f"Processor output keys: {processed_features.keys() if hasattr(processed_features, 'keys') else 'no keys'}")
                return
            
            # Move to device
            input_features = input_features.to(self.device)
            
            # Convert to half precision if model is in half precision
            if self.device == "cuda" and hasattr(self.model, 'dtype') and self.model.dtype == torch.float16:
                input_features = input_features.half()
            
            # Generate with proper settings
            with torch.no_grad():
                outputs = self.model.generate(
                    input_features,
                    max_length=448,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode the tokens to text
            transcription = self.processor.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence score (average of token probabilities)
            try:
                # Get best token probability at each step
                best_token_probs = []
                for score_set in outputs.scores:
                    token_probs = torch.softmax(score_set, dim=-1)
                    best_probs = torch.max(token_probs, dim=-1).values
                    best_token_probs.extend(best_probs.cpu().numpy())
                
                # Average as confidence score
                confidence = float(np.mean(best_token_probs)) if best_token_probs else 0.0
            except Exception as e:
                self.logger.warning(f"Error calculating confidence: {e}")
                confidence = 0.0
            
            # Update transcription results
            with self.transcription_lock:
                self.last_transcription = transcription
                self.last_confidence = confidence
            
            # Performance metrics
            elapsed = time.time() - start_time
            self.transcription_count += 1
            self.total_processing_time += elapsed
            self.total_audio_duration += buffer_duration
            
            # Log transcription results
            self.logger.info(
                f"Transcribed {buffer_duration:.1f}s audio in {elapsed:.2f}s "
                f"(conf: {confidence:.2f}): {transcription}"
            )
            
            # Optional: Write to file
            with open("transcription.txt", "a") as f:
                f.write(f"{transcription}->{confidence:.2f}\n")
                
        except Exception as e:
            self.logger.error(f"Error in transcription: {e}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
    
    def start_transcribing(self):
        """Start the transcription process"""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.logger.warning("Transcription already running")
            return
            
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Transcription processing started")
    
    def get_transcription(self):
        """Get the latest transcription with metadata"""
        with self.transcription_lock:
            return {
                "text": self.last_transcription,
                "confidence": self.last_confidence
            }
    
    def stop(self):
        """Stop the transcription process with proper cleanup"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping transcription...")
        self.is_running = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Clean up PyAudio resources
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.audio_interface:
            self.audio_interface.terminate()
            self.audio_interface = None
        
        self.logger.info("Transcription stopped and resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are properly cleaned up"""
        self.stop()
        

if __name__ == "__main__":
    # Create and start the SpeakSense system
    system = SpeakSense()
    system.start()