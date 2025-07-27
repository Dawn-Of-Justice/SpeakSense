import os
import time
import queue
import threading
import numpy as np
import pyaudio
import wave
from groq import Groq
from pydub import AudioSegment
from io import BytesIO

class RealtimeTranscriber:
    def __init__(self, api_key="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", model="distil-whisper-large-v3-en", 
                 prompt="", chunk_duration=3, sample_rate=16000, 
                 channels=1, format=pyaudio.paInt16):
        """
        Initialize the real-time transcriber.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env variable)
            model: Groq transcription model to use
            prompt: Optional prompt to improve transcription context
            chunk_duration: Duration of audio chunks to process (seconds)
            sample_rate: Audio sample rate
            channels: Number of audio channels (1=mono, 2=stereo)
            format: Audio format (pyaudio.paInt16 recommended)
        """
        self.client = Groq(api_key=api_key)
        self.model = model
        self.prompt = prompt
        
        # Audio settings
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.frames_per_chunk = int(chunk_duration * sample_rate)
        
        # Threading components
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_recording = False
        self.is_processing = False
        self.recording_thread = None
        self.processing_thread = None
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def start(self):
        """Start real-time transcription."""
        if self.is_recording:
            print("Already running!")
            return
            
        self.is_recording = True
        self.is_processing = True
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Real-time transcription started.")
        
    def stop(self):
        """Stop real-time transcription."""
        self.is_recording = False
        self.is_processing = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
        self.p.terminate()
        print("Real-time transcription stopped.")
    
    def get_transcription(self, timeout=0.1):
        """
        Get the latest transcription result if available.
        
        Returns:
            Transcription text or None if no new results
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _record_audio(self):
        """Thread function to continuously record audio."""
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        
        audio_buffer = []
        buffer_samples = 0
        target_samples = self.frames_per_chunk
        
        try:
            while self.is_recording:
                # Read audio data
                data = stream.read(1024, exception_on_overflow=False)
                audio_buffer.append(data)
                buffer_samples += 1024
                
                # If we have enough data for a chunk, process it
                if buffer_samples >= target_samples:
                    # Convert to audio segment
                    audio_data = b''.join(audio_buffer)
                    self.audio_queue.put(audio_data)
                    
                    # Reset buffer but keep some overlap
                    overlap_samples = min(int(0.5 * self.sample_rate), buffer_samples)
                    audio_buffer = audio_buffer[-int(overlap_samples/1024):]
                    buffer_samples = overlap_samples
        finally:
            stream.stop_stream()
            stream.close()
    
    def _process_audio(self):
        """Thread function to process audio chunks and get transcriptions."""
        while self.is_processing:
            try:
                # Get audio chunk from queue (wait up to 0.5 seconds)
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Save audio data to a temporary file-like object
                with BytesIO() as wav_buffer:
                    with wave.open(wav_buffer, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(self.p.get_sample_size(self.format))
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_data)
                    
                    wav_buffer.seek(0)
                    
                    # Convert to m4a (required by Groq)
                    wav_audio = AudioSegment.from_wav(wav_buffer)
                    m4a_buffer = BytesIO()
                    wav_audio.export(m4a_buffer, format="mp4")
                    m4a_buffer.seek(0)
                    
                    # Send to Groq API
                    try:
                        transcription = self.client.audio.transcriptions.create(
                            file=("audio.m4a", m4a_buffer.read()),
                            model=self.model,
                            prompt=self.prompt,
                            response_format="verbose_json",
                        )
                        
                        # Put result in output queue
                        if hasattr(transcription, 'text') and transcription.text.strip():
                            self.result_queue.put(transcription.text)
                    except Exception as e:
                        print(f"Transcription error: {e}")
                
            except queue.Empty:
                # No audio data available yet
                pass
            except Exception as e:
                print(f"Processing error: {e}")
                
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Example usage
if __name__ == "__main__":
    import time
    
    # Using context manager
    with RealtimeTranscriber(prompt="This is a tech discussion") as transcriber:
        print("Speak now! (Press Ctrl+C to stop)")
        try:
            while True:
                result = transcriber.get_transcription()
                if result:
                    print(f"Transcription: {result}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
            
    # Or manually
    """
    transcriber = RealtimeTranscriber()
    transcriber.start()
    
    try:
        while True:
            result = transcriber.get_transcription()
            if result:
                print(f"Transcription: {result}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        transcriber.stop()
    """