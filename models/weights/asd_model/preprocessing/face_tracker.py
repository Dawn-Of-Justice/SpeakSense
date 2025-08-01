import numpy as np
import cv2
import time
import threading
from collections import deque


class FaceTracker:
    """
    Advanced face tracking module that enhances the ASD system by providing
    more robust tracking, identity persistence, and additional features.
    """
    
    def __init__(self, config=None):
        """
        Initialize the face tracker with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with parameters:
                - max_faces (int): Maximum number of faces to track
                - iou_threshold (float): IOU threshold for matching faces
                - max_track_age (int): Maximum age of tracks in frames
                - min_track_len (int): Minimum track length to consider valid
                - smoothing_factor (float): EMA smoothing factor [0-1]
                - feature_tracking (bool): Whether to use feature tracking
        """
        # Default configuration
        self.config = {
            'max_faces': 5,
            'iou_threshold': 0.45,
            'max_track_age': 30,  # frames
            'min_track_len': 3,  # frames
            'smoothing_factor': 0.7,  # EMA smoothing factor
            'feature_tracking': True,  # Use feature tracking (KLT)
            'debug_mode': False,  # Print debug info
            'use_face_recognition': False,  # Use face recognition for identity
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize tracking variables
        self.tracks = []
        self.next_track_id = 0
        self.last_frame = None
        self.last_gray = None
        self.frame_index = 0
        
        # Feature tracker (for KLT)
        self.feature_tracker_params = dict(
            maxCorners=100,  # Maximum corners to track
            qualityLevel=0.3,  # Quality level for corner detection
            minDistance=7,  # Minimum distance between corners
            blockSize=7  # Block size for corner detection
        )
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Face recognition (if enabled)
        self.face_embeddings = {}
        if self.config['use_face_recognition']:
            try:
                # Try to import face_recognition library if available
                import face_recognition
                self.has_face_recognition = True
            except ImportError:
                print("Face recognition library not available, disabling feature")
                self.config['use_face_recognition'] = False
                self.has_face_recognition = False
        else:
            self.has_face_recognition = False
            
        print(f"FaceTracker initialized with config: {self.config}")
    
    def update(self, frame, detections):
        """
        Update tracks with new frame and detections.
        
        Args:
            frame: Current video frame
            detections: List of face detection dictionaries with 'bbox' and 'conf'
            
        Returns:
            list: Updated tracks
        """
        if frame is None or not detections:
            return self.tracks
            
        with self.lock:
            # Convert frame to grayscale for feature tracking
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Increment frame index
            self.frame_index += 1
            
            # Match detections to existing tracks
            matched_tracks, unmatched_detections = self._match_detections(detections)
            
            # Update matched tracks
            for track_idx, detection_idx in matched_tracks:
                self._update_track(track_idx, detections[detection_idx], frame, gray)
                
            # Create new tracks for unmatched detections
            for detection_idx in unmatched_detections:
                self._create_track(detections[detection_idx], frame, gray)
                
            # Update KLT tracking for all tracks
            if self.config['feature_tracking'] and self.last_gray is not None:
                self._update_klt_tracking(self.last_gray, gray)
                
            # Remove old tracks
            self._remove_old_tracks()
            
            # Update face recognition embeddings if enabled
            if self.config['use_face_recognition'] and self.has_face_recognition:
                self._update_face_embeddings(frame)
                
            # Update last frame and gray
            self.last_frame = frame.copy()
            self.last_gray = gray.copy()
            
            return self.tracks
    
    def get_active_tracks(self, min_len=None):
        """
        Get list of active tracks that meet minimum length requirement.
        
        Args:
            min_len (int): Minimum track length, or None to use configured value
            
        Returns:
            list: Active tracks
        """
        if min_len is None:
            min_len = self.config['min_track_len']
            
        with self.lock:
            return [t for t in self.tracks if len(t['frames']) >= min_len]
    
    def get_track_by_id(self, track_id):
        """
        Get track by its ID.
        
        Args:
            track_id (int): Track ID to find
            
        Returns:
            dict: Track data or None if not found
        """
        with self.lock:
            for track in self.tracks:
                if track['id'] == track_id:
                    return track
            return None
    
    def get_nearest_track(self, point):
        """
        Get the track nearest to a given point.
        
        Args:
            point (tuple): (x, y) point coordinates
            
        Returns:
            dict: Nearest track or None if no tracks
        """
        if not self.tracks:
            return None
            
        with self.lock:
            nearest_track = None
            min_distance = float('inf')
            
            for track in self.tracks:
                if not track['x'] or not track['y']:
                    continue
                    
                # Use latest position
                x, y = track['x'][-1], track['y'][-1]
                
                # Calculate distance
                distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_track = track
                    
            return nearest_track
    
    def get_track_features(self, track_id=None):
        """
        Get enhanced features for a specific track or all tracks.
        
        Args:
            track_id (int): Track ID or None for all tracks
            
        Returns:
            dict or list: Track features
        """
        with self.lock:
            if track_id is not None:
                # Get specific track
                track = self.get_track_by_id(track_id)
                if track:
                    return self._extract_track_features(track)
                return None
                
            # Get all active tracks
            active_tracks = self.get_active_tracks()
            return [self._extract_track_features(track) for track in active_tracks]
    
    def visualize_tracks(self, frame):
        """
        Draw tracks on frame for visualization.
        
        Args:
            frame: Video frame to draw on
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        if frame is None:
            return frame
            
        vis_frame = frame.copy()
        
        with self.lock:
            # Draw each active track
            for track in self.get_active_tracks():
                # Skip if no positions
                if not track['x'] or not track['y'] or not track['s']:
                    continue
                    
                # Get latest position
                x, y, s = track['x'][-1], track['y'][-1], track['s'][-1]
                
                # Draw bounding box
                x1, y1 = int(x - s), int(y - s)
                x2, y2 = int(x + s), int(y + s)
                
                # Color based on track ID (cycle through colors)
                color_id = track['id'] % 6
                colors = [
                    (0, 255, 0),    # Green
                    (255, 0, 0),    # Blue
                    (0, 0, 255),    # Red
                    (0, 255, 255),  # Yellow
                    (255, 0, 255),  # Magenta
                    (255, 255, 0)   # Cyan
                ]
                color = colors[color_id]
                
                # Draw rectangle
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw track ID
                cv2.putText(vis_frame, f"ID: {track['id']}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw track length
                cv2.putText(vis_frame, f"Len: {len(track['frames'])}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw feature points if available
                if 'klt_points' in track and track['klt_points'] is not None:
                    for pt in track['klt_points']:
                        if pt is not None:
                            cv2.circle(vis_frame, (int(pt[0]), int(pt[1])), 2, color, -1)
                            
        return vis_frame
    
    def _match_detections(self, detections):
        """
        Match new detections to existing tracks using IOU.
        
        Args:
            detections: List of face detection dictionaries
            
        Returns:
            tuple: (matched_tracks, unmatched_detections)
                matched_tracks: List of (track_idx, detection_idx) pairs
                unmatched_detections: List of detection indices
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections)))
            
        # Calculate IOUs between all tracks and detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            # Skip tracks without bounding boxes
            if not track['bboxes']:
                continue
                
            # Get latest bbox
            track_bbox = track['bboxes'][-1]
            
            for j, detection in enumerate(detections):
                detection_bbox = detection['bbox']
                iou = self._bbox_iou(track_bbox, detection_bbox)
                iou_matrix[i, j] = iou
        
        # Match using greedy assignment
        matched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        # For each track, find best matching detection
        for i in range(len(self.tracks)):
            if not unmatched_detections:
                break
                
            # Find best match
            best_match = -1
            best_iou = self.config['iou_threshold']
            
            for j in unmatched_detections:
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_match = j
                    
            # If found a match, add it
            if best_match >= 0:
                matched_tracks.append((i, best_match))
                unmatched_detections.remove(best_match)
                
        return matched_tracks, unmatched_detections
    
    def _update_track(self, track_idx, detection, frame, gray):
        """
        Update an existing track with new detection.
        
        Args:
            track_idx (int): Index of track to update
            detection (dict): New detection data
            frame: Current video frame
            gray: Grayscale version of frame
        """
        track = self.tracks[track_idx]
        bbox = detection['bbox']
        
        # Calculate center and size
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        s = max((bbox[3] - bbox[1]), (bbox[2] - bbox[0])) / 2
        
        # Apply smoothing if track has history
        if track['x'] and track['y'] and track['s']:
            alpha = self.config['smoothing_factor']
            x = alpha * x + (1 - alpha) * track['x'][-1]
            y = alpha * y + (1 - alpha) * track['y'][-1]
            s = alpha * s + (1 - alpha) * track['s'][-1]
        
        # Update track data
        track['frames'].append(self.frame_index)
        track['bboxes'].append(bbox)
        track['x'].append(x)
        track['y'].append(y)
        track['s'].append(s)
        track['conf'].append(detection.get('conf', 1.0))
        track['last_update'] = self.frame_index
        
        # Extract face image for recognition if enabled
        if self.config['use_face_recognition'] and self.has_face_recognition:
            face_img = self._extract_face_image(frame, bbox)
            if face_img is not None:
                track['face_img'] = face_img
        
        # Update KLT points if feature tracking enabled
        if self.config['feature_tracking']:
            # Update on every 3rd frame for efficiency
            if self.frame_index % 3 == 0:
                self._update_track_features(track, gray, bbox)
    
    def _create_track(self, detection, frame, gray):
        """
        Create a new track from detection.
        
        Args:
            detection (dict): Detection data
            frame: Current video frame
            gray: Grayscale version of frame
        """
        bbox = detection['bbox']
        
        # Calculate center and size
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
        s = max((bbox[3] - bbox[1]), (bbox[2] - bbox[0])) / 2
        
        # Create new track
        new_track = {
            'id': self.next_track_id,
            'frames': [self.frame_index],
            'bboxes': [bbox],
            'x': [x],
            'y': [y],
            's': [s],
            'conf': [detection.get('conf', 1.0)],
            'klt_points': None,
            'face_embedding': None,
            'face_img': None,
            'last_update': self.frame_index
        }
        
        # Extract face image for recognition if enabled
        if self.config['use_face_recognition'] and self.has_face_recognition:
            face_img = self._extract_face_image(frame, bbox)
            if face_img is not None:
                new_track['face_img'] = face_img
        
        # Initialize KLT points if feature tracking enabled
        if self.config['feature_tracking']:
            self._update_track_features(new_track, gray, bbox)
            
        # Add to tracks list
        self.tracks.append(new_track)
        self.next_track_id += 1
    
    def _remove_old_tracks(self):
        """
        Remove tracks that haven't been updated recently.
        """
        if not self.tracks:
            return
            
        # Calculate age threshold
        age_threshold = self.frame_index - self.config['max_track_age']
        
        # Filter tracks that have been updated recently
        self.tracks = [t for t in self.tracks if t['last_update'] >= age_threshold]
    
    def _update_klt_tracking(self, prev_gray, curr_gray):
        """
        Update KLT tracking for all tracks.
        
        Args:
            prev_gray: Previous grayscale frame
            curr_gray: Current grayscale frame
        """
        if not self.tracks:
            return
            
        for track in self.tracks:
            # Skip if no KLT points
            if 'klt_points' not in track or track['klt_points'] is None or len(track['klt_points']) == 0:
                continue
                
            # Get last known points
            p0 = np.float32([pt for pt in track['klt_points'] if pt is not None]).reshape(-1, 1, 2)
            
            # Skip if no valid points
            if p0.shape[0] == 0:
                continue
                
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, p0, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Check if points were found
            if p1 is not None and st is not None:
                # Filter good points
                good_new = p1[st==1].reshape(-1, 2)
                good_old = p0[st==1].reshape(-1, 2)
                
                # Skip if no good points
                if len(good_new) == 0:
                    continue
                    
                # Calculate median displacement
                dx = np.median(good_new[:, 0] - good_old[:, 0])
                dy = np.median(good_new[:, 1] - good_old[:, 1])
                
                # Update track position if no recent detection (gap filling)
                if track['last_update'] < self.frame_index - 2 and len(track['x']) > 0:
                    # Update position with KLT displacement
                    x = track['x'][-1] + dx
                    y = track['y'][-1] + dy
                    s = track['s'][-1]  # Keep size same
                    
                    # Add to track
                    track['frames'].append(self.frame_index)
                    track['x'].append(x)
                    track['y'].append(y)
                    track['s'].append(s)
                    
                    # Create estimated bbox
                    bbox = [x - s, y - s, x + s, y + s]
                    track['bboxes'].append(bbox)
                    track['conf'].append(0.5)  # Lower confidence for KLT
                    track['last_update'] = self.frame_index
                
                # Update KLT points
                track['klt_points'] = good_new.tolist()
    
    def _update_track_features(self, track, gray, bbox):
        """
        Update KLT feature points for a track.
        
        Args:
            track (dict): Track to update
            gray: Grayscale frame
            bbox: Bounding box [x1, y1, x2, y2]
        """
        # Extract ROI
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = gray.shape
        
        # Ensure within frame
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        
        # Skip if invalid ROI
        if x2 <= x1 or y2 <= y1:
            return
            
        # Extract ROI
        roi = gray[y1:y2, x1:x2]
        
        # Detect feature points
        points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_tracker_params)
        
        # Skip if no points detected
        if points is None:
            return
            
        # Convert to global coordinates
        if len(points) > 0:
            for i in range(len(points)):
                points[i][0][0] += x1
                points[i][0][1] += y1
                
            # Extract points
            track['klt_points'] = [pt[0].tolist() for pt in points]
        else:
            track['klt_points'] = []
    
    def _update_face_embeddings(self, frame):
        """
        Update face recognition embeddings for active tracks.
        
        Args:
            frame: Current video frame
        """
        if not self.has_face_recognition:
            return
            
        import face_recognition
        
        # Process every 5th frame for efficiency
        if self.frame_index % 5 != 0:
            return
            
        for track in self.tracks:
            # Skip if no face image or already has embedding
            if track.get('face_img') is None or track.get('face_embedding') is not None:
                continue
                
            try:
                # Get face encoding
                face_img = track['face_img']
                encodings = face_recognition.face_encodings(face_img)
                
                if encodings:
                    # Store embedding
                    track['face_embedding'] = encodings[0]
                    
                    # Store in embeddings dictionary
                    self.face_embeddings[track['id']] = encodings[0]
            except Exception as e:
                if self.config['debug_mode']:
                    print(f"Error extracting face embedding: {e}")
    
    def _extract_face_image(self, frame, bbox):
        """
        Extract face image from frame.
        
        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            numpy.ndarray: Face image or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            h, w = frame.shape[:2]
            
            # Ensure within frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            
            # Skip if invalid ROI
            if x2 <= x1 or y2 <= y1:
                return None
                
            # Extract face image
            face_img = frame[y1:y2, x1:x2]
            
            return face_img
        except Exception as e:
            if self.config['debug_mode']:
                print(f"Error extracting face image: {e}")
            return None
    
    def _bbox_iou(self, bbox1, bbox2):
        """
        Calculate IOU between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IOU value [0-1]
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate areas
        w_i = max(0, x2_i - x1_i)
        h_i = max(0, y2_i - y1_i)
        intersection = w_i * h_i
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IOU
        union = area1 + area2 - intersection
        if union <= 0:
            return 0
            
        return intersection / union
    
    def _extract_track_features(self, track):
        """
        Extract enhanced features from track.
        
        Args:
            track (dict): Track data
            
        Returns:
            dict: Enhanced features
        """
        # Skip if track doesn't have enough history
        if not track['x'] or not track['y'] or not track['s']:
            return None
            
        # Get latest position and size
        x = track['x'][-1]
        y = track['y'][-1]
        s = track['s'][-1]
        
        # Calculate movement features
        dx = dy = 0
        if len(track['x']) > 1:
            dx = track['x'][-1] - track['x'][-2]
            dy = track['y'][-1] - track['y'][-2]
            
        # Calculate size change
        ds = 0
        if len(track['s']) > 1:
            ds = track['s'][-1] - track['s'][-2]
            
        # Calculate track stability (lower value = more stable)
        stability = 0
        if len(track['x']) > 5:
            # Calculate variance of recent positions
            recent_x = track['x'][-5:]
            recent_y = track['y'][-5:]
            
            var_x = np.var(recent_x)
            var_y = np.var(recent_y)
            
            # Normalize by face size
            stability = (var_x + var_y) / (s * s)
            stability = min(1.0, stability)
            
        # Calculate track length normalized [0-1]
        track_length = min(1.0, len(track['frames']) / 30)
        
        # Calculate average confidence
        avg_conf = sum(track['conf']) / len(track['conf']) if track['conf'] else 0
        
        # Create feature dictionary
        features = {
            'track_id': track['id'],
            'position': (x, y),
            'size': s,
            'movement': (dx, dy),
            'size_change': ds,
            'stability': stability,
            'track_length': track_length,
            'confidence': avg_conf,
            'last_update': track['last_update'],
            'frame_index': self.frame_index,
            'age': self.frame_index - track['frames'][0]
        }
        
        return features


# Example usage if run directly
if __name__ == "__main__":
    import cv2
    
    # Initialize tracker
    tracker = FaceTracker({
        'debug_mode': True
    })
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open video capture
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Convert to detections format
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': [float(x), float(y), float(x+w), float(y+h)],
                    'conf': 0.9
                })
                
            # Update tracker
            tracks = tracker.update(frame, detections)
            
            # Get track features
            track_features = tracker.get_track_features()
            
            # Visualize tracks
            vis_frame = tracker.visualize_tracks(frame)
            
            # Display features
            if track_features:
                y_pos = 30
                for features in track_features:
                    if features:
                        cv2.putText(vis_frame, 
                                   f"Track {features['track_id']}: Stability: {features['stability']:.2f}, Length: {features['track_length']:.2f}",
                                   (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_pos += 20
            
            # Display the frame
            cv2.imshow('Face Tracker', vis_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()