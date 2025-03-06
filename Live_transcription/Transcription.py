import torch
import numpy as np
import pyaudio
import threading
import queue
import time
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class WhisperRealtimeTranscriber:
    def __init__(self, model_name="openai/whisper-medium", device=None):
        # Set device to CUDA if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Whisper model and processor
        print("Loading Whisper model and processor...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Set model to English-only for better performance
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="english", task="transcribe")
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
        self.buffer_max_size = self.sample_rate * 30  # 30 seconds max (Whisper can handle longer segments)
        self.last_transcription = ""
        self.silence_threshold = 0.01
        self.silence_counter = 0
        self.max_silence_chunks = 12  # About 3 seconds of silence
        
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
        print("Listening started! Speak into your microphone...")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
        
    def _process_audio(self):
        """Process audio chunks from the queue"""
        while self.is_running:
            if not self.audio_queue.empty():
                audio_chunk = self.audio_queue.get()
                
                # Convert to numpy array
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Check for silence
                if np.abs(audio_np).mean() < self.silence_threshold:
                    self.silence_counter += 1
                else:
                    self.silence_counter = 0
                
                # Add to buffer
                self.audio_buffer = np.append(self.audio_buffer, audio_np)
                
                # Truncate buffer if it gets too large
                if len(self.audio_buffer) > self.buffer_max_size:
                    self.audio_buffer = self.audio_buffer[-self.buffer_max_size:]
                
                # Process buffer if enough silence or buffer is large enough
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                if (self.silence_counter >= self.max_silence_chunks and buffer_duration > 1.0) or \
                   buffer_duration >= 5.0:  # Process every 5 seconds or after silence
                    self._transcribe_buffer()
            else:
                time.sleep(0.1)  # Prevent CPU hogging
    
    def _transcribe_buffer(self):
        """Transcribe the current audio buffer"""
        if len(self.audio_buffer) < self.sample_rate * 0.5:  # At least 0.5 seconds
            return
            
        start_time = time.time()
        
        # Process audio for Whisper input
        input_features = self.processor(
            self.audio_buffer, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate token ids
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, max_length=448)
        
        # Decode ids to text
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Measure and display transcription time
        elapsed = time.time() - start_time
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        # Clear screen and show transcription
        os.system('cls' if os.name == 'nt' else 'clear')
        print("Whisper Real-time Transcription")
        print("-" * 50)
        print(f"Audio duration: {buffer_duration:.1f}s | Processing time: {elapsed:.1f}s")
        print("-" * 50)
        print(transcription)
        print("-" * 50)
        print("Press Ctrl+C to stop")
        with open("transcription.txt", "a") as f:
            f.write(transcription+"->")
        
        # Reset buffer if significant silence detected
        if self.silence_counter >= self.max_silence_chunks:
            # Keep a small portion for context
            self.audio_buffer = self.audio_buffer[-int(self.sample_rate * 0.5):]
            self.silence_counter = 0
        else:
            # Keep the most recent 2 seconds to maintain context
            self.audio_buffer = self.audio_buffer[-int(self.sample_rate * 2):]
            
        self.last_transcription = transcription
    
    def start_transcribing(self):
        """Start the transcription process"""
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """Stop the transcription process"""
        self.is_running = False
        
        if hasattr(self, 'stream') and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
            
        self.audio_interface.terminate()
        print("Transcription stopped.")


def main():
    # Create the Whisper transcriber
    transcriber = WhisperRealtimeTranscriber()
    
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

if __name__ == "__main__":
    main()