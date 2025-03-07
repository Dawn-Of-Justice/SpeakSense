import cv2
import numpy as np
import time
import math


class VisualFeatureExtractor:
    """
    Extracts enhanced visual features from processed frames and face detections
    for the context-aware RNN model.
    """
    
    def __init__(self, config=None):
        """
        Initialize the visual feature extractor with configuration parameters.
        
        Args:
            config (dict): Configuration parameters including:
                - feature_dimensions (int): Number of output features (default: 10)
                - use_head_pose (bool): Whether to extract head pose features (default: True)
                - use_eye_gaze (bool): Whether to extract eye gaze features (default: True)
                - use_attention_features (bool): Whether to extract attention features (default: True)
                - debug_mode (bool): Whether to enable debug visualization (default: False)
        """
        # Default configuration
        self.config = {
            'feature_dimensions': 10,
            'use_head_pose': True,
            'use_eye_gaze': True,
            'use_attention_features': True,
            'debug_mode': False,
            'face_size_threshold': 50,  # Min face size in pixels for reliable feature extraction
            'smoothing_factor': 0.7,  # EMA smoothing factor for feature stability (0-1)
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize feature history for smoothing
        self.feature_history = {}
        
        # Initialize face landmark detector (if available)
        try:
            # Try to load facial landmark detector from OpenCV's face module
            self.face_landmark_detector = cv2.face.createFacemarkLBF()
            model_path = "models/lbfmodel.yaml"  # Path to pre-trained model
            try:
                self.face_landmark_detector.loadModel(model_path)
                self.has_landmarks = True
                print(f"Facial landmark detector loaded from {model_path}")
            except Exception as e:
                print(f"Error loading facial landmark model: {e}")
                self.has_landmarks = False
        except:
            print("Facial landmark detection not available, using simplified feature extraction")
            self.has_landmarks = False
            
        # Initialize eye detector as fallback
        self.eye_cascade = None
        try:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            print("Eye detector loaded successfully")
        except Exception as e:
            print(f"Error loading eye detector: {e}")
        
        print(f"VisualFeatureExtractor initialized with config: {self.config}")
    
    def extract_features(self, frame, face_data):
        """
        Extract visual features from a frame and detected face.
        
        Args:
            frame: The video frame
            face_data: Face detection data with bbox [x1, y1, x2, y2] and confidence
            
        Returns:
            dict: Dictionary of extracted features
        """
        if frame is None or face_data is None:
            return None
            
        try:
            # Extract bounding box
            bbox = face_data.get('bbox', None)
            if bbox is None or len(bbox) != 4:
                return None
                
            # Ensure bbox is in correct format [x1, y1, x2, y2]
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Calculate face center and size
            face_center_x = (x1 + x2) / 2
            face_center_y = (y1 + y2) / 2
            face_width = x2 - x1
            face_height = y2 - y1
            face_size = max(face_width, face_height)
            
            # Skip if face is too small for reliable feature extraction
            if face_size < self.config['face_size_threshold']:
                return {
                    'track_id': face_data.get('track_id', -1),
                    'face_center': (face_center_x, face_center_y),
                    'face_size': face_size,
                    'normalized_center_x': face_center_x / frame.shape[1],
                    'normalized_center_y': face_center_y / frame.shape[0],
                    'normalized_size': face_size / max(frame.shape[0], frame.shape[1]),
                    'feature_quality': 0.2,  # Low quality for small faces
                    'head_pose': [0, 0, 0],
                    'eye_gaze': [0, 0],
                    'attention_score': 0.3,  # Default low attention
                }
                
            # Crop face region for detailed processing
            # Ensure crop is within frame boundaries
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            face_crop = frame[y1:y2, x1:x2]
            
            # Initialize feature dictionary
            features = {
                'track_id': face_data.get('track_id', -1),
                'face_center': (face_center_x, face_center_y),
                'face_size': face_size,
                'normalized_center_x': face_center_x / frame.shape[1],
                'normalized_center_y': face_center_y / frame.shape[0],
                'normalized_size': face_size / max(frame.shape[0], frame.shape[1]),
                'feature_quality': 0.8,  # Default good quality
            }
            
            # Extract facial landmarks if available
            landmarks = None
            if self.has_landmarks:
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                success, landmarks = self.face_landmark_detector.fit(gray, np.array([[0, 0, face_width, face_height]]))
                if success:
                    landmarks = landmarks[0]  # Get the first face's landmarks
            
            # Extract head pose features
            if self.config['use_head_pose']:
                head_pose = self._extract_head_pose(face_crop, landmarks)
                features['head_pose'] = head_pose
            else:
                features['head_pose'] = [0, 0, 0]  # Default neutral pose
            
            # Extract eye gaze features
            if self.config['use_eye_gaze']:
                eye_gaze = self._extract_eye_gaze(face_crop, landmarks)
                features['eye_gaze'] = eye_gaze
            else:
                features['eye_gaze'] = [0, 0]  # Default neutral gaze
            
            # Extract attention features
            if self.config['use_attention_features']:
                attention_score = self._calculate_attention_score(features['head_pose'], features['eye_gaze'], 
                                                                face_center_x, face_center_y, frame.shape[1], frame.shape[0])
                features['attention_score'] = attention_score
            else:
                features['attention_score'] = 0.5  # Default neutral attention
            
            # Apply smoothing using exponential moving average
            track_id = features['track_id']
            if track_id not in self.feature_history:
                self.feature_history[track_id] = features.copy()
            else:
                alpha = self.config['smoothing_factor']
                # Smooth continuous values
                for key in ['normalized_center_x', 'normalized_center_y', 'normalized_size', 
                           'attention_score', 'feature_quality']:
                    if key in features and key in self.feature_history[track_id]:
                        features[key] = alpha * features[key] + (1-alpha) * self.feature_history[track_id][key]
                
                # Smooth arrays
                for key in ['head_pose', 'eye_gaze']:
                    if key in features and key in self.feature_history[track_id]:
                        for i in range(len(features[key])):
                            features[key][i] = alpha * features[key][i] + (1-alpha) * self.feature_history[track_id][key][i]
                
                # Update history
                self.feature_history[track_id] = features.copy()
            
            return features
            
        except Exception as e:
            print(f"Error extracting visual features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _extract_head_pose(self, face_crop, landmarks=None):
        """
        Extract head pose estimation features.
        
        Returns:
            list: [pitch, yaw, roll] in normalized range [-1, 1]
        """
        # Default neutral pose
        pitch, yaw, roll = 0.0, 0.0, 0.0
        
        try:
            if landmarks is not None and len(landmarks) >= 68:
                # Use landmarks for more accurate pose estimation
                # This is a simplified approach - in production, use a dedicated pose estimator
                
                # Get key points for pose estimation
                nose_tip = landmarks[30]
                chin = landmarks[8]
                left_eye = landmarks[36]
                right_eye = landmarks[45]
                left_mouth = landmarks[48]
                right_mouth = landmarks[54]
                
                # Calculate yaw (left-right) from eye midpoint horizontal position relative to nose
                eye_midpoint_x = (left_eye[0] + right_eye[0]) / 2
                yaw = (eye_midpoint_x - nose_tip[0]) / (face_crop.shape[1] / 4)
                yaw = max(-1.0, min(1.0, yaw))  # Normalize to [-1, 1]
                
                # Calculate pitch (up-down) from nose position relative to face center
                face_center_y = face_crop.shape[0] / 2
                pitch = (face_center_y - nose_tip[1]) / (face_crop.shape[0] / 4)
                pitch = max(-1.0, min(1.0, pitch))  # Normalize to [-1, 1]
                
                # Calculate roll (tilt) from eye angle
                if right_eye[0] != left_eye[0]:  # Avoid division by zero
                    eye_angle = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
                    roll = eye_angle / (math.pi / 4)  # Normalize to approx [-1, 1]
                    roll = max(-1.0, min(1.0, roll))
            else:
                # Fallback to simpler heuristics
                # Use face position in frame as approximate yaw indicator
                h, w = face_crop.shape[:2]
                face_center_x = w / 2
                face_center_y = h / 2
                
                # Convert gray and detect edges for feature extraction
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                
                # Use horizontal gradient for yaw approximation
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                abs_sobelx = np.absolute(sobelx)
                left_energy = np.sum(abs_sobelx[:, :w//2])
                right_energy = np.sum(abs_sobelx[:, w//2:])
                yaw = (right_energy - left_energy) / max(right_energy + left_energy, 1)
                yaw = max(-1.0, min(1.0, yaw))
                
                # Use vertical gradient for pitch approximation
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                abs_sobely = np.absolute(sobely)
                top_energy = np.sum(abs_sobely[:h//2, :])
                bottom_energy = np.sum(abs_sobely[h//2:, :])
                pitch = (bottom_energy - top_energy) / max(bottom_energy + top_energy, 1)
                pitch = max(-1.0, min(1.0, pitch))
                
                # Approximate roll from overall image orientation
                roll = 0.0  # Default to no roll in simplified model
        
        except Exception as e:
            print(f"Error in head pose estimation: {e}")
            pitch, yaw, roll = 0.0, 0.0, 0.0
            
        return [pitch, yaw, roll]
    
    def _extract_eye_gaze(self, face_crop, landmarks=None):
        """
        Extract eye gaze direction features.
        
        Returns:
            list: [horizontal_gaze, vertical_gaze] in normalized range [-1, 1]
        """
        # Default neutral gaze
        h_gaze, v_gaze = 0.0, 0.0
        
        try:
            if landmarks is not None and len(landmarks) >= 68:
                # Use facial landmarks for more accurate gaze estimation
                
                # Get eye landmarks
                left_eye_points = landmarks[36:42]
                right_eye_points = landmarks[42:48]
                
                # Calculate eye centers
                left_eye_center = np.mean(left_eye_points, axis=0)
                right_eye_center = np.mean(right_eye_points, axis=0)
                
                # Calculate eye sizes
                left_eye_width = np.linalg.norm(left_eye_points[0] - left_eye_points[3])
                right_eye_width = np.linalg.norm(right_eye_points[0] - right_eye_points[3])
                
                # Extract eye regions for pupil detection
                left_eye_region = self._extract_eye_region(face_crop, left_eye_points)
                right_eye_region = self._extract_eye_region(face_crop, right_eye_points)
                
                # Detect pupils
                left_pupil = self._detect_pupil(left_eye_region)
                right_pupil = self._detect_pupil(right_eye_region)
                
                # Calculate gaze direction from pupil position
                if left_pupil is not None and right_pupil is not None:
                    # Normalize pupil positions relative to eye centers
                    left_h_offset = (left_pupil[0] - left_eye_region.shape[1]/2) / (left_eye_width / 2)
                    right_h_offset = (right_pupil[0] - right_eye_region.shape[1]/2) / (right_eye_width / 2)
                    
                    left_v_offset = (left_pupil[1] - left_eye_region.shape[0]/2) / (left_eye_region.shape[0] / 2)
                    right_v_offset = (right_pupil[1] - right_eye_region.shape[0]/2) / (right_eye_region.shape[0] / 2)
                    
                    # Average the offsets from both eyes
                    h_gaze = (left_h_offset + right_h_offset) / 2
                    v_gaze = (left_v_offset + right_v_offset) / 2
                    
                    # Clip to reasonable range
                    h_gaze = max(-1.0, min(1.0, h_gaze))
                    v_gaze = max(-1.0, min(1.0, v_gaze))
            else:
                # Fallback to eye detector if landmarks not available
                if self.eye_cascade is not None:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(eyes) >= 2:
                        # Sort eyes by x-coordinate (left to right)
                        eyes = sorted(eyes, key=lambda x: x[0])
                        
                        # Simple approach: compare eye positions to face center
                        face_center_x = face_crop.shape[1] / 2
                        face_center_y = face_crop.shape[0] / 2
                        
                        # Calculate eye centers
                        eye_centers = []
                        for (ex, ey, ew, eh) in eyes:
                            eye_center_x = ex + ew/2
                            eye_center_y = ey + eh/2
                            eye_centers.append((eye_center_x, eye_center_y))
                        
                        if len(eye_centers) >= 2:
                            left_eye, right_eye = eye_centers[0], eye_centers[1]
                            
                            # Calculate horizontal gaze based on eye position relative to face center
                            left_offset = (left_eye[0] - face_crop.shape[1]/3) / (face_crop.shape[1]/6)
                            right_offset = (right_eye[0] - 2*face_crop.shape[1]/3) / (face_crop.shape[1]/6)
                            
                            h_gaze = (left_offset + right_offset) / 2
                            h_gaze = max(-1.0, min(1.0, h_gaze))
                            
                            # Calculate vertical gaze
                            v_left = (left_eye[1] - face_crop.shape[0]/2) / (face_crop.shape[0]/4)
                            v_right = (right_eye[1] - face_crop.shape[0]/2) / (face_crop.shape[0]/4)
                            v_gaze = (v_left + v_right) / 2
                            v_gaze = max(-1.0, min(1.0, v_gaze))
        
        except Exception as e:
            print(f"Error in eye gaze estimation: {e}")
            h_gaze, v_gaze = 0.0, 0.0
            
        return [h_gaze, v_gaze]
    
    def _extract_eye_region(self, face_image, eye_points):
        """
        Extract the eye region from face image using eye landmark points.
        """
        try:
            # Convert points to numpy array
            eye_points = np.array(eye_points, dtype=np.int32)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(eye_points)
            
            # Add margin
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(face_image.shape[1] - x, w + 2*margin)
            h = min(face_image.shape[0] - y, h + 2*margin)
            
            # Extract region
            eye_region = face_image[y:y+h, x:x+w]
            
            # Resize for consistent processing
            eye_region = cv2.resize(eye_region, (60, 40))
            
            return eye_region
        except Exception as e:
            print(f"Error extracting eye region: {e}")
            # Return empty region as fallback
            return np.zeros((40, 60, 3), dtype=np.uint8)
    
    def _detect_pupil(self, eye_region):
        """
        Detect pupil in eye region using image processing techniques.
        
        Returns:
            tuple: (x, y) pupil center or None if detection fails
        """
        try:
            if eye_region.size == 0:
                return None
                
            # Convert to grayscale
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better contrast
            gray_eye = cv2.equalizeHist(gray_eye)
            
            # Apply Gaussian blur
            gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
            
            # Use adaptive thresholding to identify dark regions (pupil)
            _, threshold = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
                
            # Select largest contour as pupil
            pupil_contour = max(contours, key=cv2.contourArea)
            
            # Get moments to find center
            M = cv2.moments(pupil_contour)
            
            if M["m00"] == 0:
                return None
                
            # Calculate pupil center
            pupil_x = int(M["m10"] / M["m00"])
            pupil_y = int(M["m01"] / M["m00"])
            
            return (pupil_x, pupil_y)
            
        except Exception as e:
            print(f"Error detecting pupil: {e}")
            return None
    
    def _calculate_attention_score(self, head_pose, eye_gaze, face_x, face_y, frame_width, frame_height):
        """
        Calculate attention score based on head pose, eye gaze, and face position.
        
        Returns:
            float: Attention score in range [0, 1] where 1 means high attention to camera
        """
        try:
            # Extract components
            pitch, yaw, roll = head_pose
            h_gaze, v_gaze = eye_gaze
            
            # Normalize face position to [-1, 1]
            norm_face_x = (face_x / frame_width) * 2 - 1
            norm_face_y = (face_y / frame_height) * 2 - 1
            
            # Calculate attention components
            
            # 1. Head orientation factor (higher when looking straight at camera)
            # Yaw is most important for determining if facing camera
            head_factor = 1.0 - abs(yaw) * 0.5 - abs(pitch) * 0.3 - abs(roll) * 0.2
            head_factor = max(0.0, min(1.0, head_factor))
            
            # 2. Eye gaze factor (higher when eyes looking at camera)
            gaze_factor = 1.0 - abs(h_gaze) * 0.6 - abs(v_gaze) * 0.4
            gaze_factor = max(0.0, min(1.0, gaze_factor))
            
            # 3. Position factor (higher when face is centered)
            pos_factor = 1.0 - (abs(norm_face_x) * 0.3 + abs(norm_face_y) * 0.3)
            pos_factor = max(0.0, min(1.0, pos_factor))
            
            # Combine factors with weights
            attention_score = head_factor * 0.5 + gaze_factor * 0.4 + pos_factor * 0.1
            
            return attention_score
            
        except Exception as e:
            print(f"Error calculating attention score: {e}")
            return 0.5  # Default neutral attention
    
    def create_feature_vector(self, features):
        """
        Create a normalized feature vector for the RNN model.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            numpy.ndarray: Feature vector of specified dimensions
        """
        if features is None:
            return np.zeros(self.config['feature_dimensions'])
            
        try:
            # Create a standard feature vector with the most important features
            # This ensures consistency regardless of which features were extracted
            feature_vector = []
            
            # Position features (normalized)
            feature_vector.append(features.get('normalized_center_x', 0.5))
            feature_vector.append(features.get('normalized_center_y', 0.5))
            feature_vector.append(features.get('normalized_size', 0.3))
            
            # Head pose (pitch, yaw, roll)
            head_pose = features.get('head_pose', [0, 0, 0])
            feature_vector.extend(head_pose)
            
            # Eye gaze (horizontal, vertical)
            eye_gaze = features.get('eye_gaze', [0, 0])
            feature_vector.extend(eye_gaze)
            
            # Attention score
            feature_vector.append(features.get('attention_score', 0.5))
            
            # Quality indicator
            feature_vector.append(features.get('feature_quality', 0.5))
            
            # Pad or truncate to match required dimensions
            if len(feature_vector) < self.config['feature_dimensions']:
                feature_vector.extend([0] * (self.config['feature_dimensions'] - len(feature_vector)))
            elif len(feature_vector) > self.config['feature_dimensions']:
                feature_vector = feature_vector[:self.config['feature_dimensions']]
                
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"Error creating feature vector: {e}")
            return np.zeros(self.config['feature_dimensions'])
    
    def visualize_features(self, frame, features):
        """
        Create a visualization of extracted features for debugging.
        
        Args:
            frame: The original video frame
            features: Dictionary of extracted features
            
        Returns:
            numpy.ndarray: Frame with visualization overlays
        """
        if not self.config['debug_mode'] or frame is None or features is None:
            return frame
            
        try:
            # Make a copy of the frame
            vis_frame = frame.copy()
            
            # Get face center and size
            face_center = features.get('face_center', None)
            face_size = features.get('face_size', 0)
            
            if face_center and face_size > 0:
                x, y = face_center
                s = face_size / 2
                
                # Draw face box
                cv2.rectangle(vis_frame, 
                             (int(x-s), int(y-s)), 
                             (int(x+s), int(y+s)), 
                             (0, 255, 0), 2)
                
                # Draw head pose arrow
                head_pose = features.get('head_pose', [0, 0, 0])
                pitch, yaw, roll = head_pose
                
                # Use yaw for horizontal direction
                arrow_length = int(face_size * 0.5)
                arrow_x = int(x + yaw * arrow_length)
                arrow_y = int(y + pitch * arrow_length * 0.5)
                
                cv2.arrowedLine(vis_frame, 
                               (int(x), int(y)), 
                               (arrow_x, arrow_y), 
                               (0, 0, 255), 2)
                
                # Draw eye gaze
                eye_gaze = features.get('eye_gaze', [0, 0])
                h_gaze, v_gaze = eye_gaze
                
                gaze_x = int(x + h_gaze * face_size * 0.3)
                gaze_y = int(y + v_gaze * face_size * 0.3)
                
                cv2.circle(vis_frame, (gaze_x, gaze_y), 5, (255, 0, 0), -1)
                
                # Show attention score
                attention_score = features.get('attention_score', 0)
                cv2.putText(vis_frame, 
                           f"Attention: {attention_score:.2f}", 
                           (int(x-s), int(y-s-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 255), 1)
                
                # Show quality
                quality = features.get('feature_quality', 0)
                cv2.putText(vis_frame, 
                           f"Quality: {quality:.2f}", 
                           (int(x-s), int(y-s-30)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 255), 1)
            
            return vis_frame
            
        except Exception as e:
            print(f"Error in feature visualization: {e}")
            return frame


# Example usage if run directly
if __name__ == "__main__":
    # This requires a video source to test
    from video_processor import VideoProcessor
    
    # Initialize processors
    video_proc = VideoProcessor({
        'input_source': 0,  # Default camera
        'frame_width': 640,
        'frame_height': 480,
        'fps': 30
    })
    
    feature_ext = VisualFeatureExtractor({
        'debug_mode': True  # Enable visualization
    })
    
    # Simplified face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if video_proc.start():
        try:
            # Create a window
            cv2.namedWindow("Feature Extraction Test", cv2.WINDOW_NORMAL)
            
            track_id = 0  # Simple ID for demo
            
            while True:
                # Get processed frame
                frame, metadata = video_proc.get_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Detect faces (simplified for demo)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.1, 4)
                
                # Process first face found
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    
                    # Create face data structure similar to ASD output
                    face_data = {
                        'track_id': track_id,
                        'bbox': [float(x), float(y), float(x+w), float(y+h)],
                        'conf': 0.9
                    }
                    
                    # Extract features
                    features = feature_ext.extract_features(frame, face_data)
                    
                    if features:
                        # Create feature vector
                        feature_vector = feature_ext.create_feature_vector(features)
                        
                        # Display feature vector on frame
                        cv2.putText(frame, 
                                   f"Features: {feature_vector[:5]}...", 
                                   (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (0, 255, 255), 1)
                        
                        # Visualize features
                        frame = feature_ext.visualize_features(frame, features)
                
                # Display the frame
                cv2.imshow("Feature Extraction Test", frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            video_proc.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start video processor")