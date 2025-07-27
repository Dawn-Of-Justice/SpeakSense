import time
import numpy as np
import cv2
import threading
import queue

# Import the realtime ASD implementation
from .realtime_asd import RealtimeASD


class ASDWrapper:
    """
    Wrapper class for the RealtimeASD component that provides enhanced feature extraction
    and a clean interface for the context-aware RNN.
    """
    
    def __init__(self, config=None):
        """
        Initialize the ASD wrapper with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with parameters:
                - model_path (str): Path to the ASD model weights
                - use_cuda (bool): Whether to use CUDA acceleration
                - feature_smoothing (float): EMA smoothing factor [0-1]
                - attention_threshold (float): Threshold for considering someone attentive
                - speaking_threshold (float): Threshold for considering someone speaking
                - debug_mode (bool): Enable debug information
        """
        # Default configuration
        self.config = {
            'model_path': 'weight/pretrain_AVA_CVPR.model',
            'use_cuda': False,
            'feature_smoothing': 0.7,
            'attention_threshold': 0.6,
            'speaking_threshold': 2.0,
            'debug_mode': False
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize the underlying ASD model
        self.asd = RealtimeASD(
            model_path=self.config['model_path'],
            use_cuda=self.config['use_cuda']
        )
        
        # Feature history for temporal smoothing
        self.feature_history = {}
        
        # Internal state
        self.last_update_time = time.time()
        self.is_running = False
        
        # Results queue for thread safety
        self.results_queue = queue.Queue(maxsize=5)
        
        # Lock for thread safety
        self.state_lock = threading.Lock()
        
        print(f"ASDWrapper initialized with config: {self.config}")
    
    def start(self):
        """
        Start the ASD processing.
        """
        if self.is_running:
            print("ASD wrapper is already running")
            return
            
        # Start the underlying ASD system
        self.asd.start()
        self.is_running = True
        
        # Start the feature extraction thread
        self.feature_thread = threading.Thread(target=self._feature_extraction_loop)
        self.feature_thread.daemon = True
        self.feature_thread.start()
        
        print("ASD wrapper started")
        return True
    
    def stop(self):
        """
        Stop the ASD processing.
        """
        self.is_running = False
        
        # Stop the underlying ASD system
        self.asd.stop()
        
        # Wait for thread to finish
        if hasattr(self, 'feature_thread'):
            self.feature_thread.join(timeout=1.0)
            
        print("ASD wrapper stopped")
    
    def process_frame(self, frame):
        """
        Process a video frame.
        
        Args:
            frame: Video frame to process
        """
        if frame is None:
            return
            
        # Pass to underlying ASD system
        self.asd.add_frame(frame)
    
    def process_audio(self, audio_chunk):
        """
        Process an audio chunk.
        
        Args:
            audio_chunk: Audio data to process
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return
            
        # Pass to underlying ASD system
        self.asd.add_audio(audio_chunk)
    
    def get_features(self):
        """
        Get enhanced features from ASD for each detected face.
        
        Returns:
            list: List of feature dictionaries for each track
        """
        try:
            # Non-blocking check for new results
            if self.results_queue.empty():
                return []
                
            # Get latest results
            features = self.results_queue.get_nowait()
            return features
            
        except queue.Empty:
            return []
    
    def get_face_count(self):
        """
        Get the number of faces currently being tracked.
        
        Returns:
            int: Number of active face tracks
        """
        return len(self.asd.tracks) if hasattr(self.asd, 'tracks') else 0
    
    def is_speaking_to_robot(self, track_id=None):
        """
        Simple heuristic to determine if a person is likely speaking to the robot
        based only on ASD features (without the RNN).
        
        Args:
            track_id: ID of the track to check, or None for best candidate
            
        Returns:
            tuple: (is_speaking_to_robot, confidence, track_id)
        """
        features = self.get_features()
        
        if not features:
            return False, 0.0, -1
            
        # If track_id specified, check only that track
        if track_id is not None:
            for feature in features:
                if feature.get('track_id') == track_id:
                    is_speaking = feature.get('is_speaking', False)
                    is_looking = feature.get('is_looking', False)
                    attention = feature.get('attention_score', 0.0)
                    
                    # Simple heuristic: speaking AND looking with good attention
                    if is_speaking and is_looking and attention > self.config['attention_threshold']:
                        confidence = attention * 0.7 + 0.3
                        return True, confidence, track_id
                    else:
                        confidence = max(0.0, attention - 0.3)
                        return False, confidence, track_id
            
            return False, 0.0, track_id
            
        # No track specified, find best candidate
        best_confidence = 0.0
        best_track_id = -1
        is_best_speaking = False
        
        for feature in features:
            is_speaking = feature.get('is_speaking', False)
            is_looking = feature.get('is_looking', False)
            attention = feature.get('attention_score', 0.0)
            
            # Calculate confidence score
            if is_speaking and is_looking:
                confidence = attention * 0.7 + 0.3
            else:
                confidence = max(0.0, attention - 0.3)
                
            # Keep track of best candidate
            if confidence > best_confidence:
                best_confidence = confidence
                best_track_id = feature.get('track_id', -1)
                is_best_speaking = is_speaking and is_looking and attention > self.config['attention_threshold']
                
        return is_best_speaking, best_confidence, best_track_id
    
    def _feature_extraction_loop(self):
        """
        Background thread for continuous feature extraction.
        """
        while self.is_running:
            try:
                # Get latest ASD results
                results = self.asd.get_results()
                
                if not results:
                    time.sleep(0.01)  # Short sleep if no results
                    continue
                    
                # Process each result (only using the latest one)
                if results:
                    display_frame = results[-1]
                    self._extract_enhanced_features(display_frame)
                    
                # Short sleep to prevent CPU spinning
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Error in feature extraction: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)  # Longer sleep after error
    
    def _extract_enhanced_features(self, frame):
        """
        Extract enhanced features from ASD results.
        
        Args:
            frame: The latest display frame with ASD results
        """
        try:
            # List to hold enhanced features for each track
            enhanced_features = []
            
            # Process each active track
            for i, track in enumerate(self.asd.tracks):
                # Skip tracks without enough history
                if len(track.get('frames', [])) < 2:
                    continue
                    
                # Get the ASD score for this track
                score = 0.0
                is_speaking = False
                is_looking = False
                
                if i in self.asd.track_scores and self.asd.track_scores[i]['scores']:
                    # The original ASD model returns scores that we've mapped to three states:
                    # 2.6 = speaking, 0.6 = looking, -0.6 = not looking
                    score = self.asd.track_scores[i]['scores'][-1]
                    
                    # Determine state based on score
                    is_speaking = abs(score - 2.6) < 0.5  # Speaking state
                    is_looking = abs(score - 0.6) < 0.5  # Looking state
                
                # Get face position and size
                if not track.get('x', []) or not track.get('y', []) or not track.get('s', []):
                    continue
                    
                face_x = track['x'][-1]
                face_y = track['y'][-1]
                face_size = track['s'][-1]
                
                # Calculate frame center 
                frame_height, frame_width = frame.shape[:2]
                frame_center_x = frame_width / 2
                frame_center_y = frame_height / 2
                
                # Calculate normalized distance from center (0 = center, 1 = edge)
                distance_from_center = np.sqrt(
                    ((face_x - frame_center_x) / frame_width) ** 2 +
                    ((face_y - frame_center_y) / frame_height) ** 2
                ) * 2  # Scale to [0, 1] range
                
                distance_from_center = min(1.0, distance_from_center)
                
                # Calculate face size relative to frame
                relative_size = min(1.0, face_size / (min(frame_width, frame_height) / 4))
                
                # Calculate attention score based on looking state and position
                attention_score = 0.0
                
                if is_looking:
                    # Higher attention when looking directly and centered
                    attention_score = 0.7 - (distance_from_center * 0.4)
                    # Bonus for larger faces (closer to camera)
                    attention_score += relative_size * 0.3
                elif is_speaking:
                    # If speaking but not looking, moderate attention
                    attention_score = 0.4 - (distance_from_center * 0.2)
                else:
                    # Not looking or speaking, low attention
                    attention_score = 0.1
                    
                # Ensure in [0, 1] range
                attention_score = max(0.0, min(1.0, attention_score))
                
                # Create feature dictionary
                features = {
                    'track_id': i,
                    'is_speaking': is_speaking,
                    'is_looking': is_looking,
                    'raw_score': score,
                    'face_position': [face_x / frame_width, face_y / frame_height],  # Normalized
                    'face_size': relative_size,
                    'distance_from_center': distance_from_center,
                    'attention_score': attention_score,
                    'timestamp': time.time()
                }
                
                # Apply temporal smoothing
                smoothed_features = self._apply_smoothing(features)
                enhanced_features.append(smoothed_features)
            
            # Update the results queue (clear old results first)
            while self.results_queue.qsize() > 0:
                try:
                    self.results_queue.get_nowait()
                except queue.Empty:
                    break
                    
            self.results_queue.put(enhanced_features)
            
        except Exception as e:
            print(f"Error extracting enhanced features: {e}")
            import traceback
            traceback.print_exc()
    
    def _apply_smoothing(self, features):
        """
        Apply exponential moving average smoothing to feature values.
        
        Args:
            features: Feature dictionary to smooth
            
        Returns:
            dict: Smoothed features
        """
        track_id = features['track_id']
        
        # If no history for this track, initialize with current features
        if track_id not in self.feature_history:
            self.feature_history[track_id] = features.copy()
            return features
            
        # Apply EMA smoothing to continuous values
        alpha = self.config['feature_smoothing']
        
        # Smoothable features
        smoothable = [
            'attention_score', 
            'distance_from_center',
            'face_size'
        ]
        
        # Array features
        array_features = [
            'face_position'
        ]
        
        # Create smoothed copy
        smoothed = features.copy()
        
        # Smooth scalar values
        for key in smoothable:
            if key in features and key in self.feature_history[track_id]:
                smoothed[key] = alpha * features[key] + (1-alpha) * self.feature_history[track_id][key]
                
        # Smooth array values
        for key in array_features:
            if key in features and key in self.feature_history[track_id]:
                for i in range(len(features[key])):
                    smoothed[key][i] = alpha * features[key][i] + (1-alpha) * self.feature_history[track_id][key][i]
        
        # Update history with new smoothed values
        self.feature_history[track_id] = smoothed.copy()
        
        return smoothed
    
    def get_feature_vector(self, track_id=None):
        """
        Get a feature vector for the given track suitable for RNN input.
        
        Args:
            track_id: ID of track to get features for, or None for best candidate
            
        Returns:
            numpy.ndarray: Feature vector
        """
        features = self.get_features()
        
        if not features:
            return np.zeros(10)  # Default empty feature vector
            
        # Find the specified track or best candidate
        target_features = None
        
        if track_id is not None:
            # Find the specified track
            for feature in features:
                if feature.get('track_id') == track_id:
                    target_features = feature
                    break
        else:
            # Find the track with highest attention score
            best_score = -1
            for feature in features:
                score = feature.get('attention_score', 0)
                if score > best_score:
                    best_score = score
                    target_features = feature
        
        if target_features is None:
            return np.zeros(10)  # No suitable track found
            
        # Create feature vector
        vector = [
            float(target_features.get('is_speaking', False)),
            float(target_features.get('is_looking', False)),
            target_features.get('attention_score', 0.0),
            target_features.get('face_position', [0.5, 0.5])[0],  # x position
            target_features.get('face_position', [0.5, 0.5])[1],  # y position
            target_features.get('face_size', 0.3),
            target_features.get('distance_from_center', 0.5),
            target_features.get('raw_score', 0.0) / 3.0,  # Normalize
        ]
        
        # Pad to consistent length
        while len(vector) < 10:
            vector.append(0.0)
            
        return np.array(vector, dtype=np.float32)


# Example usage if run directly
if __name__ == "__main__":
    import cv2
    
    # Initialize the wrapper
    asd_wrapper = ASDWrapper({
        'use_cuda': False,
        'debug_mode': True
    })
    
    # Start the wrapper
    asd_wrapper.start()
    
    try:
        # Open video capture
        cap = cv2.VideoCapture(0)
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            asd_wrapper.process_frame(frame)
            
            # Generate simple audio (in reality, you would capture audio)
            dummy_audio = np.random.normal(0, 0.01, 1600).astype(np.float32)
            asd_wrapper.process_audio(dummy_audio)
            
            # Get features
            features = asd_wrapper.get_features()
            
            # Update display
            if features:
                for feature in features:
                    # Get track info
                    track_id = feature.get('track_id', -1)
                    is_speaking = feature.get('is_speaking', False)
                    is_looking = feature.get('is_looking', False)
                    attention = feature.get('attention_score', 0.0)
                    
                    # Print status
                    status = []
                    if is_speaking:
                        status.append("SPEAKING")
                    if is_looking:
                        status.append("LOOKING")
                        
                    print(f"Track {track_id}: {', '.join(status) if status else 'NEUTRAL'}, "
                         f"Attention: {attention:.2f}")
                    
                    # Check if speaking to robot
                    is_addressing, conf, tid = asd_wrapper.is_speaking_to_robot(track_id)
                    if is_addressing:
                        print(f"Track {tid} is ADDRESSING ROBOT with confidence {conf:.2f}")
            
            # Display the frame
            cv2.imshow('ASD Wrapper Demo', frame)
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        asd_wrapper.stop()
        cv2.destroyAllWindows()