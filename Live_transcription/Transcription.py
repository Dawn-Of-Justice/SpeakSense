import torch
import numpy as np
import pyaudio
import threading
import queue
import time
import os
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WhisperTranscriber")

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
        # Set device to CUDA if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model and processor
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
            
            # Set model to English-only for better performance
            self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="english", 
                task="transcribe"
            )
            
            # Optimize model if requested and on GPU
            if optimize and self.device == "cuda":
                self.model = self.model.half()  # Use FP16 for faster inference
                logger.info("Using half precision (FP16) for faster inference")
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            logger.error(f"Error initializing PyAudio: {e}")
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
            logger.warning("Already listening")
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
            logger.info("Audio capture started - listening to microphone")
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.stop()
            raise
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        if status:
            logger.debug(f"Audio callback status: {status}")
            
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
                logger.debug(f"Recalibrated silence threshold to {self.silence_threshold:.5f}")
                
            # Reset calibration state
            self.calibration_samples = self.calibration_samples[-10:]  # Keep some history
            self.calibration_counter = 0
    
    def _process_audio(self):
        """Process audio chunks from the queue and manage transcription"""
        logger.info("Audio processing thread started")
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
                        logger.debug(f"Silence detected ({self.silence_counter}/{self.max_silence_chunks})")
                    else:
                        if self.silence_counter > 0:
                            logger.debug("Audio activity resumed")
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
                logger.error(f"Error in audio processing: {e}")
                time.sleep(0.1)  # Avoid tight loop on error
    
    def _transcribe_buffer(self):
        """Transcribe the current audio buffer"""
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
            logger.debug("Buffer too small, skipping transcription")
            return
        
        try:
            # Measure transcription performance
            start_time = time.time()
            buffer_duration = len(audio_to_process) / self.sample_rate
            
            # Process audio for Whisper input
            input_features = self.processor(
                audio_to_process, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate with token probabilities for confidence scoring
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
                logger.warning(f"Error calculating confidence: {e}")
                confidence = 0.0
            
            # Try to get word-level timestamps if model supports it
            timestamps = None
            try:
                if hasattr(self.processor, "batch_decode_timestamps"):
                    timestamps = self.processor.batch_decode_timestamps(
                        outputs.sequences, outputs.alignment_heads
                    )
            except:
                pass  # Ignore if timestamps not supported
            
            # Update transcription results
            with self.transcription_lock:
                self.last_transcription = transcription
                self.last_confidence = confidence
                self.last_timestamps = timestamps
            
            # Performance metrics
            elapsed = time.time() - start_time
            self.transcription_count += 1
            self.total_processing_time += elapsed
            self.total_audio_duration += buffer_duration
            
            # Log transcription results
            logger.info(
                f"Transcribed {buffer_duration:.1f}s audio in {elapsed:.2f}s "
                f"(conf: {confidence:.2f}): {transcription}"
            )
            
            # Optional: Write to file
            with open("transcription.txt", "a") as f:
                f.write(f"{transcription}->{confidence:.2f}\n")
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
    
    def start_transcribing(self):
        """Start the transcription process"""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            logger.warning("Transcription already running")
            return
            
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Transcription processing started")
    
    def get_transcription(self):
        """Get the latest transcription with metadata"""
        with self.transcription_lock:
            return {
                "text": self.last_transcription,
                "confidence": self.last_confidence,
                "timestamps": self.last_timestamps
            }
    
    def get_performance_metrics(self):
        """Get performance metrics for the transcription system"""
        if self.transcription_count == 0:
            return {
                "avg_processing_time": 0,
                "realtime_factor": 0,
                "transcription_count": 0
            }
            
        avg_time = self.total_processing_time / self.transcription_count
        realtime_factor = self.total_processing_time / max(self.total_audio_duration, 0.001)
        
        return {
            "avg_processing_time": avg_time,
            "realtime_factor": realtime_factor,
            "transcription_count": self.transcription_count
        }
    
    def stop(self):
        """Stop the transcription process with proper cleanup"""
        if not self.is_running:
            return
            
        logger.info("Stopping transcription...")
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
        
        logger.info("Transcription stopped and resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are properly cleaned up"""
        self.stop()


def main():
    """Run the transcriber as a standalone application"""
    # Create the Whisper transcriber
    transcriber = WhisperRealtimeTranscriber()
    
    try:
        # Start listening and transcribing
        transcriber.start_listening()
        transcriber.start_transcribing()
        
        # Keep running until Ctrl+C
        print("Press Ctrl+C to stop transcription")
        while True:
            result = transcriber.get_transcription()
            print(f"Latest: {result['text']} (Confidence: {result['confidence']:.2f})")
            
            # Show performance metrics periodically
            if transcriber.transcription_count > 0 and transcriber.transcription_count % 5 == 0:
                metrics = transcriber.get_performance_metrics()
                print(f"Performance: RT factor: {metrics['realtime_factor']:.2f}x, "
                      f"Avg time: {metrics['avg_processing_time']:.2f}s")
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        transcriber.stop()

if __name__ == "__main__":
    main()