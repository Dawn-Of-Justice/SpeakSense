import whisper
import numpy as np
import pyaudio
import threading
import queue
import time
import os

class WhisperRealtimeTranscriber:
    def __init__(self, model_name="tiny", device=None):
        # Set device to CUDA if available
        self.device = device if device else ("cuda" if whisper.torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Whisper model
        print("Loading Whisper model...")
        self.model = whisper.load_model(model_name, device=self.device)
        print("Model loaded!")
        
        # Audio configurations
        self.sample_rate = 16000  # Hz
        self.chunk_size = 4000    # Samples per chunk
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.audio_interface = pyaudio.PyAudio()
        
        # Create a queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Tracking for continuous transcription
        self.is_running = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_max_size = self.sample_rate * 30  # 30 seconds max
        self.last_transcription = ""
        
        # Enhanced VAD settings
        self.silence_threshold = None  # Will be auto-calibrated
        self.silence_counter = 0
        self.max_silence_chunks = 15   # About 3-4 seconds of silence
        
        # Auto-calibration variables
        self.calibration_samples = []
        self.is_calibrated = False
        self.calibration_frames = 60  # Calibrate over 60 frames
        
    def auto_calibrate_silence_threshold(self, audio_energy):
        """Auto-calibrate silence threshold based on ambient noise"""
        if not self.is_calibrated:
            self.calibration_samples.append(audio_energy)
            
            # Show calibration progress
            if len(self.calibration_samples) % 15 == 0:
                print(f"🔧 Calibrating... ({len(self.calibration_samples)}/{self.calibration_frames}) Current energy: {audio_energy:.4f}")
            
            if len(self.calibration_samples) >= self.calibration_frames:
                # Calculate baseline noise and set threshold
                baseline_noise = np.mean(self.calibration_samples)
                noise_std = np.std(self.calibration_samples)
                
                # Set threshold as baseline + 2 standard deviations for better detection
                self.silence_threshold = baseline_noise + (2 * noise_std)
                self.is_calibrated = True
                
                print(f"✅ Auto-calibration completed!")
                print(f"📊 Baseline noise: {baseline_noise:.4f}")
                print(f"📈 Noise std dev: {noise_std:.4f}")
                print(f"🎛️  Speech threshold: {self.silence_threshold:.4f}")
                print("🎤 Ready for speech detection!")
                print("-" * 50)
        
    def start_listening(self):
        """Start capturing audio from microphone"""
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
        print("🎯 Whisper Real-time Transcriber Started!")
        print("🔧 Calibrating microphone noise levels...")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
        
    def _process_audio(self):
        """Process audio chunks from the queue with enhanced VAD"""
        while self.is_running:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                
                # Convert to numpy array
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Calculate audio energy (RMS)
                audio_energy = np.sqrt(np.mean(audio_np ** 2))
                
                # Auto-calibrate threshold if not done
                if not self.is_calibrated:
                    self.auto_calibrate_silence_threshold(audio_energy)
                    continue
                
                # Enhanced VAD logic
                is_speech = audio_energy > self.silence_threshold
                
                if is_speech:
                    if self.silence_counter > 0:
                        print(f"🔴 Speech detected! (energy: {audio_energy:.4f})")
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1
                
                # Add to buffer
                self.audio_buffer = np.append(self.audio_buffer, audio_np)
                
                # Truncate buffer if it gets too large
                if len(self.audio_buffer) > self.buffer_max_size:
                    self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]
                
                # Process buffer with enhanced conditions
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                should_transcribe = (
                    # After sufficient silence and minimum speech duration
                    (self.silence_counter >= self.max_silence_chunks and buffer_duration > 1.5) or
                    # Or when buffer gets too long (every 8 seconds max)
                    buffer_duration >= 8.0 or
                    # Or when we detect end of longer speech (more silence frames for longer speech)
                    (buffer_duration > 5.0 and self.silence_counter >= self.max_silence_chunks // 2)
                )
                
                if should_transcribe:
                    self._transcribe_buffer()
                    
            else:
                time.sleep(0.01)  # Prevent CPU hogging
                
    def _transcribe_buffer(self):
        """Transcribe the current audio buffer with enhanced logic"""
        if len(self.audio_buffer) < self.sample_rate * 0.8:  # At least 0.8 seconds
            # Reset buffer if too short
            self.audio_buffer = np.array([], dtype=np.float32)
            self.silence_counter = 0
            return
            
        start_time = time.time()
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        print(f"🔄 Transcribing {buffer_duration:.1f}s of audio...")
        
        # Use the native Whisper library for transcription
        result = self.model.transcribe(
            self.audio_buffer,
            language="en",  # Specify English for better results
            fp16=False if self.device == "cpu" else True,  # Use FP16 precision on GPU
            no_speech_threshold=0.6,  # Reduces hallucinations on silence
            initial_prompt=self.last_transcription[-100:] if self.last_transcription else None,  # Add context
        )
        
        transcription = result["text"].strip()
        elapsed = time.time() - start_time
        
        if transcription:  # Only show non-empty transcriptions
            print(f"🎤 Transcription: {transcription}")
            print(f"⏱️  Processing time: {elapsed:.2f}s")
            
            # Save to file
            with open("transcription.txt", "a", encoding="utf-8") as f:
                f.write(f"{transcription}\n")
                
            self.last_transcription += " " + transcription
        else:
            print("🔇 No speech detected in buffer")
        
        # Smart buffer management based on silence detection
        if self.silence_counter >= self.max_silence_chunks:
            # Clear most of buffer after silence, keep small context
            context_size = int(self.sample_rate * 0.5)  # 0.5 seconds context
            self.audio_buffer = self.audio_buffer[-context_size:] if len(self.audio_buffer) > context_size else np.array([], dtype=np.float32)
            self.silence_counter = 0
            print("⏹️  Speech segment ended\n" + "="*50)
        else:
            # Keep recent 2 seconds for context in continuous speech
            context_size = int(self.sample_rate * 2)
            self.audio_buffer = self.audio_buffer[-context_size:]
    
    def start_transcribing(self):
        """Start the transcription process"""
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """Stop the transcription process"""
        self.is_running = False
        
        # Process any remaining buffer
        if len(self.audio_buffer) >= self.sample_rate * 0.8:
            print("🔄 Processing final audio segment...")
            self._transcribe_buffer()
            time.sleep(2)
        
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            
        self.audio_interface.terminate()
        print("✅ Transcription stopped.")


def main():
    # Create a Whisper transcriber with the "tiny" model
    # Options: "tiny", "base", "small", "medium", "large"
    transcriber = WhisperRealtimeTranscriber(model_name="tiny")
    
    try:
        # Start listening and transcribing
        transcriber.start_listening()
        transcriber.start_transcribing()
        
        # Keep running until Ctrl+C
        print("\n🛑 Press Ctrl+C to stop transcription")
        while True:
            time.sleep(5)
            if transcriber.is_calibrated and transcriber.last_transcription:
                print(f"\n📝 Recent transcription: ...{transcriber.last_transcription[-150:]}")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping transcription...")
    finally:
        transcriber.stop()

if __name__ == "__main__":
    main()