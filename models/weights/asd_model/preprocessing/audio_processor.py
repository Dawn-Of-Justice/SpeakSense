import numpy as np
import queue
import threading
import time
import math
import python_speech_features


class AudioProcessor:
    """
    Processes audio for the Active Speaker Detection model.
    Handles preprocessing, feature extraction, and VAD.
    """
    
    def __init__(self, config=None):
        """
        Initialize the audio processor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with parameters:
                - sample_rate (int): Audio sample rate in Hz
                - buffer_duration (float): Duration of audio buffer in seconds
                - window_size (int): Window size for feature extraction
                - hop_length (int): Hop length for feature extraction
                - n_mfcc (int): Number of MFCC coefficients
                - vad_threshold (float): VAD energy threshold [0-1]
        """
        # Default configuration
        self.config = {
            'sample_rate': 16000,  # Hz
            'buffer_duration': 1.0,  # seconds
            'window_size': 400,  # samples (25ms at 16kHz)
            'hop_length': 160,  # samples (10ms at 16kHz)
            'n_mfcc': 13,  # MFCC features
            'vad_threshold': 0.5,  # VAD threshold
            'max_buffer_size': 16000,  # Maximum buffer size (1 sec at 16kHz)
            'use_threading': True,  # Whether to use threaded processing
            'feature_type': 'mfcc'  # 'mfcc', 'melspectrogram', 'raw'
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Calculate buffer size in samples
        self.buffer_size = int(self.config['sample_rate'] * self.config['buffer_duration'])
        
        # Initialize audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Features and results
        self.current_features = None
        self.vad_state = False
        self.last_vad_update = time.time()
        self.speech_energy = 0.0
        
        # Queues for threading
        self.audio_queue = queue.Queue(maxsize=20)
        self.feature_queue = queue.Queue(maxsize=5)
        
        # Thread synchronization
        self.buffer_lock = threading.Lock()
        self.running = False
        
        # For energy normalization
        self.energy_history = []
        self.max_energy = 0.01  # Initial non-zero value
        self.min_energy = 0.0
        
        print(f"AudioProcessor initialized with config: {self.config}")
    
    def start(self):
        """
        Start the audio processor.
        """
        if self.running:
            print("Audio processor is already running")
            return
            
        self.running = True
        
        # Start processing thread if threading enabled
        if self.config['use_threading']:
            self.process_thread = threading.Thread(target=self._processing_loop)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            print("Audio processor started with threading")
        else:
            print("Audio processor started without threading")
            
        return True
    
    def stop(self):
        """
        Stop the audio processor.
        """
        self.running = False
        
        # Wait for thread to finish
        if self.config['use_threading'] and hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
            
        print("Audio processor stopped")
    
    def add_audio(self, audio_data):
        """
        Add audio data to the processing buffer.
        
        Args:
            audio_data: Audio data as numpy array or list
        """
        try:
            # Convert to numpy array if not already
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
                
            # Add to queue if threading, otherwise process directly
            if self.config['use_threading']:
                try:
                    # Clear queue if full
                    if self.audio_queue.full():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                            
                    self.audio_queue.put(audio_data, block=False)
                except queue.Full:
                    pass  # Skip if queue is full
            else:
                self._process_audio(audio_data)
                
        except Exception as e:
            print(f"Error adding audio: {e}")
    
    def get_features(self):
        """
        Get the latest audio features.
        
        Returns:
            dict: Feature dictionary with 'mfcc', 'energy', 'vad', etc.
        """
        if self.config['use_threading']:
            try:
                # Get latest features from queue without waiting
                if not self.feature_queue.empty():
                    self.current_features = self.feature_queue.get_nowait()
            except queue.Empty:
                pass
                
        return self.current_features
    
    def is_speech_active(self):
        """
        Check if speech is currently active based on VAD.
        
        Returns:
            bool: True if speech is active
        """
        return self.vad_state
    
    def get_speech_energy(self):
        """
        Get the current normalized speech energy level.
        
        Returns:
            float: Normalized energy level [0-1]
        """
        return self.speech_energy
    
    def _processing_loop(self):
        """
        Main processing loop for threaded operation.
        """
        while self.running:
            try:
                # Get audio from queue
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    self._process_audio(audio_data)
                except queue.Empty:
                    time.sleep(0.01)
                    continue
                    
            except Exception as e:
                print(f"Error in audio processing loop: {e}")
                time.sleep(0.1)
    
    def _process_audio(self, audio_data):
        """
        Process audio data: update buffer and extract features.
        
        Args:
            audio_data: Audio data chunk
        """
        try:
            # Update buffer with lock for thread safety
            with self.buffer_lock:
                self.audio_buffer = np.append(self.audio_buffer, audio_data)
                
                # Keep buffer at maximum size
                if len(self.audio_buffer) > self.config['max_buffer_size']:
                    self.audio_buffer = self.audio_buffer[-self.config['max_buffer_size']:]
            
            # Extract features
            features = self._extract_features()
            
            # Update VAD state
            self._update_vad(features)
            
            # Add to feature queue if threading enabled
            if self.config['use_threading']:
                # Clear queue if full
                if self.feature_queue.full():
                    try:
                        self.feature_queue.get_nowait()
                    except queue.Empty:
                        pass
                        
                self.feature_queue.put(features, block=False)
            else:
                self.current_features = features
                
        except Exception as e:
            print(f"Error processing audio: {e}")
    
    def _extract_features(self):
        """
        Extract audio features from the current buffer.
        
        Returns:
            dict: Feature dictionary
        """
        try:
            # Ensure buffer has data
            with self.buffer_lock:
                if len(self.audio_buffer) < 400:  # Need at least 25ms at 16kHz
                    return {
                        'mfcc': np.zeros((1, self.config['n_mfcc'])),
                        'energy': 0.0,
                        'normalized_energy': 0.0,
                        'vad': False,
                        'timestamp': time.time()
                    }
                    
                # Create a copy of the buffer for processing
                audio_data = self.audio_buffer.copy()
            
            # Calculate energy
            energy = np.mean(np.abs(audio_data))
            
            # Update energy history for normalization
            self.energy_history.append(energy)
            if len(self.energy_history) > 100:
                self.energy_history.pop(0)
                
            # Update min/max energy (with smoothing)
            if energy > self.max_energy:
                self.max_energy = 0.95 * self.max_energy + 0.05 * energy
            else:
                self.max_energy = 0.999 * self.max_energy + 0.001 * energy
                
            if energy < self.min_energy or self.min_energy == 0:
                self.min_energy = 0.95 * self.min_energy + 0.05 * energy
            else:
                self.min_energy = 0.999 * self.min_energy + 0.001 * energy
                
            # Normalize energy
            energy_range = max(0.001, self.max_energy - self.min_energy)
            normalized_energy = (energy - self.min_energy) / energy_range
            normalized_energy = max(0.0, min(1.0, normalized_energy))
            
            # Save for external access
            self.speech_energy = normalized_energy
            
            # Extract different feature types based on configuration
            if self.config['feature_type'] == 'mfcc':
                # Extract MFCC features
                mfcc = python_speech_features.mfcc(
                    audio_data, 
                    samplerate=self.config['sample_rate'],
                    winlen=self.config['window_size'] / self.config['sample_rate'],
                    winstep=self.config['hop_length'] / self.config['sample_rate'],
                    numcep=self.config['n_mfcc'],
                    nfilt=26,
                    nfft=512,
                    lowfreq=0,
                    highfreq=None,
                    preemph=0.97,
                    ceplifter=22,
                    appendEnergy=True
                )
                
                # Feature dictionary
                features = {
                    'mfcc': mfcc,
                    'energy': energy,
                    'normalized_energy': normalized_energy,
                    'vad': False,  # Will be updated by VAD
                    'timestamp': time.time()
                }
                
            elif self.config['feature_type'] == 'melspectrogram':
                # Extract mel spectrogram (if available)
                try:
                    # Use python_speech_features to get filterbank features (mel spectrogram)
                    fbank_feat = python_speech_features.logfbank(
                        audio_data,
                        samplerate=self.config['sample_rate'],
                        winlen=self.config['window_size'] / self.config['sample_rate'],
                        winstep=self.config['hop_length'] / self.config['sample_rate'],
                        nfilt=26,
                        nfft=512,
                        lowfreq=0,
                        highfreq=None,
                        preemph=0.97
                    )
                    
                    features = {
                        'melspectrogram': fbank_feat,
                        'energy': energy,
                        'normalized_energy': normalized_energy,
                        'vad': False,
                        'timestamp': time.time()
                    }
                    
                except Exception as e:
                    print(f"Error extracting mel spectrogram: {e}")
                    # Fall back to MFCC
                    features = {
                        'mfcc': python_speech_features.mfcc(
                            audio_data, 
                            samplerate=self.config['sample_rate'],
                            numcep=self.config['n_mfcc']
                        ),
                        'energy': energy,
                        'normalized_energy': normalized_energy,
                        'vad': False,
                        'timestamp': time.time()
                    }
                    
            else:
                # Just use raw audio with energy
                features = {
                    'raw_audio': audio_data[-1600:],  # Last 100ms at 16kHz
                    'energy': energy,
                    'normalized_energy': normalized_energy,
                    'vad': False,
                    'timestamp': time.time()
                }
                
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return {
                'mfcc': np.zeros((1, self.config['n_mfcc'])),
                'energy': 0.0,
                'normalized_energy': 0.0,
                'vad': False,
                'timestamp': time.time()
            }
    
    def _update_vad(self, features):
        """
        Update Voice Activity Detection state based on extracted features.
        
        Args:
            features: Feature dictionary
        """
        try:
            # Get energy from features
            energy = features.get('normalized_energy', 0.0)
            
            # Compare to threshold
            is_speech = energy > self.config['vad_threshold']
            
            # Apply temporal smoothing for stability
            current_time = time.time()
            time_since_update = current_time - self.last_vad_update
            
            # State transition logic:
            # - Quick to activate (100ms)
            # - Slower to deactivate (300ms)
            if is_speech and not self.vad_state:
                # Potential activation
                if energy > self.config['vad_threshold'] * 1.2:  # Higher threshold for activation
                    self.vad_state = True
                    self.last_vad_update = current_time
            elif not is_speech and self.vad_state:
                # Potential deactivation - require longer time below threshold
                if time_since_update > 0.3:  # 300ms of low energy to deactivate
                    self.vad_state = False
                    self.last_vad_update = current_time
                    
            # Update the feature dictionary
            features['vad'] = self.vad_state
            
        except Exception as e:
            print(f"Error updating VAD: {e}")
            
    def get_asd_input_features(self):
        """
        Get features formatted specifically for the ASD model.
        
        Returns:
            tuple: (feature_matrix, is_speech)
        """
        features = self.get_features()
        
        if features is None:
            # Default empty features
            return np.zeros((1, self.config['n_mfcc'])), False
            
        # Extract MFCC or appropriate features
        if 'mfcc' in features:
            feature_matrix = features['mfcc']
        elif 'melspectrogram' in features:
            feature_matrix = features['melspectrogram']
        elif 'raw_audio' in features:
            # Simple feature extraction if we only have raw audio
            # This is a fallback and won't work as well as proper MFCCs
            chunk_size = 400  # 25ms at 16kHz
            hop_size = 160    # 10ms at 16kHz
            audio = features['raw_audio']
            
            # Create simple frame-level features
            frames = []
            for i in range(0, len(audio) - chunk_size, hop_size):
                frame = audio[i:i+chunk_size]
                # Simple features: energy, zero-crossing rate
                energy = np.mean(frame**2)
                zcr = np.sum(np.abs(np.diff(np.signbit(frame)))) / len(frame)
                frames.append([energy, zcr])
                
            feature_matrix = np.array(frames) if frames else np.zeros((1, 2))
        else:
            # Last resort default
            feature_matrix = np.zeros((1, self.config['n_mfcc']))
            
        return feature_matrix, features.get('vad', False)


# Example usage if run directly
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    # Create processor
    processor = AudioProcessor({
        'sample_rate': 16000,
        'feature_type': 'mfcc'
    })
    
    # Start processor
    processor.start()
    
    # For visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # For MFCC display
    mfcc_display = ax1.imshow(np.zeros((13, 20)), aspect='auto', vmin=-5, vmax=5, origin='lower')
    ax1.set_title('MFCC Features')
    ax1.set_ylabel('Coefficients')
    ax1.set_xlabel('Frames')
    
    # For energy/VAD display
    energy_line, = ax2.plot([], [], 'b-', label='Energy')
    threshold_line, = ax2.plot([], [], 'r--', label='Threshold')
    vad_display = ax2.axvspan(0, 0, color='g', alpha=0.3, label='VAD')
    ax2.set_ylim(0, 1.1)
    ax2.set_xlim(0, 100)
    ax2.set_title('Energy and VAD')
    ax2.set_ylabel('Normalized Energy')
    ax2.set_xlabel('Frames')
    ax2.legend()
    
    # Data for plots
    energy_data = []
    vad_data = []
    frame_count = [0]  # Using a mutable object to track frame count
    
    def update_plot(frame):
        # Access frame_count through the list
        
        # Generate some audio (in a real scenario, this would come from a microphone)
        # Simulate different audio patterns:
        if frame % 100 < 30:  # Speech
            audio = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000) * 0.5
            audio += np.random.normal(0, 0.1, 1600)
        elif frame % 100 < 50:  # Silence
            audio = np.random.normal(0, 0.01, 1600)
        else:  # Background noise
            audio = np.random.normal(0, 0.05, 1600)
            
        # Process audio
        processor.add_audio(audio)
        
        # Get features
        features = processor.get_features()
        
        if features is not None:
            # Update MFCC display
            if 'mfcc' in features and features['mfcc'].shape[0] > 0:
                mfcc = features['mfcc']
                # Take last 20 frames or pad with zeros
                if mfcc.shape[0] < 20:
                    padded = np.zeros((20, mfcc.shape[1]))
                    padded[-mfcc.shape[0]:] = mfcc
                    mfcc = padded
                else:
                    mfcc = mfcc[-20:]
                    
                mfcc_display.set_array(mfcc.T)
                
            # Update energy and VAD display
            energy = features.get('normalized_energy', 0)
            vad = features.get('vad', False)
            
            energy_data.append(energy)
            vad_data.append(vad)
            
            # Keep fixed length
            if len(energy_data) > 100:
                energy_data.pop(0)
                vad_data.pop(0)
                
            # Update energy line
            energy_line.set_data(range(len(energy_data)), energy_data)
            
            # Update threshold line
            threshold_line.set_data([0, 100], [processor.config['vad_threshold'], processor.config['vad_threshold']])
            
            # Update VAD display
            vad_regions = []
            start = None
            for i, v in enumerate(vad_data):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    vad_regions.append((start, i))
                    start = None
            
            if start is not None:
                vad_regions.append((start, len(vad_data)))
                
            # Clear previous VAD highlighting
            for coll in ax2.collections:
                if isinstance(coll, plt.matplotlib.collections.PolyCollection):
                    coll.remove()
                    
            # Add new VAD highlights
            for start, end in vad_regions:
                ax2.axvspan(start, end, color='g', alpha=0.3)
            
            frame_count[0] += 1
            
        return mfcc_display, energy_line, threshold_line
    
    # Create animation
    ani = FuncAnimation(fig, update_plot, interval=100, blit=False)
    plt.tight_layout()
    plt.show()
    
    # Stop processor when done
    processor.stop()