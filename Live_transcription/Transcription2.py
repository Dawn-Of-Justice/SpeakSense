import torch
import torchaudio
import pyaudio
import numpy as np
import threading
import queue
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time

transcription = None

class RealTimeTranscriber:
    def __init__(self, model_name="facebook/wav2vec2-large-960h-lv60-self", device=None):
        # Set device to CUDA if available
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        print("Loading model and processor...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        print("Model loaded!")
        
        # Audio configurations
        self.sample_rate = 16000  # Hz
        self.chunk_size = 4000  # Samples per chunk
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Initialize PyAudio
        self.audio_interface = pyaudio.PyAudio()
        
        # Create a queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Tracking for continuous transcription
        self.is_running = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_max_size = self.sample_rate * 5  # 5 seconds max
        self.last_transcription = ""
        self.silence_threshold = 0.01
        self.silence_counter = 0
        self.max_silence_chunks = 10  # About 2.5 seconds of silence
        
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
        # print("\033[H\033[J")
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
                if (self.silence_counter >= self.max_silence_chunks and len(self.audio_buffer) > self.sample_rate * 0.5) or \
                   len(self.audio_buffer) >= self.sample_rate * 3:
                    self._transcribe_buffer()
            else:
                time.sleep(0.1)  # Prevent CPU hogging
    
    def _transcribe_buffer(self):
        global transcription
        """Transcribe the current audio buffer"""
        if len(self.audio_buffer) < self.sample_rate * 0.5:  # At least 0.5 seconds
            return
            
        # Process audio for model input
        inputs = self.processor(
            torch.tensor(self.audio_buffer), 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get logits from model
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Decode prediction
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        
        # Clear screen and show transcription
        # print("\033[H\033[J")  # Clear console
        # print("Transcription:")
        # print("-" * 50)
        # time.sleep(1)
        # print(transcription, end = " -- ")
        # time.sleep(0.1)
        # print("-" * 50)
        if transcription is not None:
            with open("transcription.txt", "a") as f:
                f.write(transcription+"->")
        
        # Reset buffer if silence detected
        if self.silence_counter >= self.max_silence_chunks:
            self.audio_buffer = np.array([], dtype=np.float32)
            self.silence_counter = 0
        else:
            # Keep last second of audio to maintain context
            self.audio_buffer = self.audio_buffer[-self.sample_rate:]
            
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
    global transcription
    # Create transcriber
    transcriber = RealTimeTranscriber()
    
    try:
        # Start listening and transcribing
        transcriber.start_listening()
        transcriber.start_transcribing()
        
        # Keep running until Ctrl+C
        print("Press Ctrl+C to stop transcription")
        while True:
            if transcription is not None:
                with open("Transcript.txt", "a") as f:
                    f.write(transcription+"--")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        transcriber.stop()

if __name__ == "__main__":
    main()