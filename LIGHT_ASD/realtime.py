import os
import time
import numpy as np
import cv2
import torch
import queue
import threading
import python_speech_features
from scipy.interpolate import interp1d
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

# Import the face detector and ASD model
# In LightASD/realtime.py
from model.faceDetector.s3fd import S3FD  # '.' means "current package" (ASD_BASED_ARCH)
from ASD import ASD                       # '.' refers to ASD.py in the same directory
# LIGHT_ASD\weight\finetuning_TalkSet.model
class RealtimeASD:
    def __init__(self, model_path="weight/finetuning_TalkSet.model", use_cuda=False):
        # Configuration parameters
        self.facedet_scale = 0.3  # Reduced for faster detection
        self.min_face_size = 1
        self.crop_scale = 0.40
        self.window_size = 15  # Significantly reduced buffer size for better performance
        self.fps = 30  # Target higher FPS
        self.min_track_len = 2  # Further reduced for quicker evaluation
        self.iou_threshold = 0.45  # Slightly reduced threshold for better face tracking
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Device selection
        self.device = 'cuda' if self.use_cuda else 'cpu'
        print(f"Using device: {self.device}")
        
        # Demo mode flag - if True, will generate simulated scores for testing
        self.demo_mode = True  # Set to True to enable simulated speaking behavior
        
        try:
            # Initialize models
            print("Initializing face detector...")
            self.face_detector = S3FD(device=self.device)
            
            print("Loading ASD model...")
            # Initialize ASD model with the appropriate device
            self.asd_model = ASD(device=self.device)
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"WARNING: Model file not found at {model_path}")
                print("Using a placeholder model for demonstration")
                self.demo_mode = True
            else:
                try:
                    self.asd_model.loadParameters(model_path)
                    print("Model loaded successfully")
                except Exception as e:
                    print(f"Error loading model parameters: {e}")
                    print("Using demo mode for speaking simulation")
                    self.demo_mode = True
            self.asd_model.eval()
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            print("Creating placeholder detection functionality")
            self.face_detector = None
            self.asd_model = None
            self.demo_mode = True
        
        # Initialize tracking variables
        self.frame_buffer = []  # Buffer to store recent frames
        self.audio_buffer = np.array([], dtype=np.float32)  # Buffer to store recent audio
        self.faces_buffer = []  # Buffer to store face detections
        self.tracks = []        # Active face tracks
        self.track_scores = {}  # Scores for each track
        self.debug_mode = False  # Set to False to reduce console output
        
        # Store speaking state for demo mode
        self.demo_speaking_states = {}
        
        # Threading resources
        self.frame_queue = queue.Queue(maxsize=10)  # Reduced for better performance
        self.audio_queue = queue.Queue(maxsize=1600)  # ~1 second of audio at 16kHz
        self.result_queue = queue.Queue(maxsize=5)
        self.running = False
        
        # Locking for thread safety
        self.buffer_lock = threading.Lock()
        
        # Performance monitoring
        self.last_frame_time = time.time()
        self.fps_display = 0
        self.frame_times = []
        
        print("Real-time ASD system initialized")
    
    def start(self):
        """Start the processing threads"""
        self.running = True
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print("Processing threads started")
    
    def stop(self):
        """Stop all processing threads"""
        self.running = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
        print("Processing stopped")
    
    def add_frame(self, frame):
        """Add a new video frame to the processing queue"""
        try:
            if self.frame_queue.qsize() > 5:  # If backed up, skip some frames
                try:
                    self.frame_queue.get_nowait()  # Remove oldest frame
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame, block=False)
            
            # Update FPS calculation
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            if elapsed > 0:
                self.frame_times.append(elapsed)
                if len(self.frame_times) > 10:
                    self.frame_times.pop(0)
                self.fps_display = 1.0 / (sum(self.frame_times) / len(self.frame_times))
            
            return True
        except queue.Full:
            return False
    
    def add_audio(self, audio_chunk):
        """Add a new audio chunk to the processing queue"""
        # Convert to numpy array if not already
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        
        # Add to circular buffer
        with self.buffer_lock:
            self.audio_buffer = np.append(self.audio_buffer, audio_chunk)
            # Keep only the most recent ~1 second (16000 samples)
            if len(self.audio_buffer) > 16000:
                self.audio_buffer = self.audio_buffer[-16000:]
    
    def get_results(self):
        """Get the latest processing results"""
        results = []
        try:
            # Get the most recent result without waiting
            if not self.result_queue.empty():
                results.append(self.result_queue.get_nowait())
        except queue.Empty:
            pass
        return results
    
    def _detect_faces(self, frame):
        """Detect faces in a frame - balanced for detection quality and speed"""
        try:
            if self.face_detector is None:
                # Fallback face detection using OpenCV's built-in detector if model wasn't loaded
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # More accurate parameters
                
                faces = []
                for (x, y, w, h) in detected_faces:
                    # Convert to [x1, y1, x2, y2] format and add confidence
                    faces.append({'bbox': [float(x), float(y), float(x+w), float(y+h)], 'conf': 0.9})
                return faces
            
            # Resize for detection - balance between speed and accuracy
            scale_factor = 0.5  # Back to original value for better detection
            small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                                    interpolation=cv2.INTER_AREA)  # Better quality
            
            # Use the S3FD detector with standard confidence threshold
            image_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            bboxes = self.face_detector.detect_faces(image_rgb, conf_th=0.8, scales=[self.facedet_scale])
            
            # Process all detected faces to ensure we don't miss any
            faces = []
            for bbox in bboxes:  # Process all faces for better detection
                try:
                    # Scale bounding box back to original frame size
                    scaled_bbox = [float(b / scale_factor) for b in bbox[:-1]]
                    # Use scaled_bbox directly - no tolist() method
                    faces.append({'bbox': scaled_bbox, 'conf': float(bbox[-1])})
                except Exception as e:
                    print(f"Error processing bbox: {e}, bbox type: {type(bbox)}")
                    continue
            
            if self.debug_mode and len(faces) > 0:
                print(f"Detected {len(faces)} faces")
                
            return faces
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def _update_tracks(self, frame_idx, faces):
        """Update face tracks with new detections"""
        # Skip if no faces detected
        if not faces:
            return
            
        # For existing tracks, find matching faces
        unmatched_faces = faces.copy()
        
        for track in self.tracks:
            if len(track['frames']) == 0 or frame_idx - track['frames'][-1] > 5:
                continue  # Skip tracks that are too old
            
            # Find the best matching face by IOU
            best_iou = -1
            best_face = None
            best_idx = -1
            
            for i, face in enumerate(unmatched_faces):
                iou = self._bb_intersection_over_union(
                    track['bboxes'][-1], face['bbox'])
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_face = face
                    best_idx = i
            
            # If found a match, update the track
            if best_face:
                track['frames'].append(frame_idx)
                track['bboxes'].append(best_face['bbox'])
                track['x'].append((best_face['bbox'][0] + best_face['bbox'][2]) / 2)
                track['y'].append((best_face['bbox'][1] + best_face['bbox'][3]) / 2)
                track['s'].append(max((best_face['bbox'][3] - best_face['bbox'][1]), 
                                   (best_face['bbox'][2] - best_face['bbox'][0])) / 2)
                
                # Apply smoothing to track parameters with exponential moving average
                if len(track['s']) > 3:
                    alpha = 0.7  # Weight for current value
                    track['s'][-1] = alpha * track['s'][-1] + (1-alpha) * track['s'][-2]
                    track['x'][-1] = alpha * track['x'][-1] + (1-alpha) * track['x'][-2]
                    track['y'][-1] = alpha * track['y'][-1] + (1-alpha) * track['y'][-2]
                
                # Remove the matched face
                if best_idx < len(unmatched_faces):
                    unmatched_faces.pop(best_idx)
        
        # Create new tracks for unmatched faces
        for face in unmatched_faces:
            x = (face['bbox'][0] + face['bbox'][2]) / 2
            y = (face['bbox'][1] + face['bbox'][3]) / 2
            s = max((face['bbox'][3] - face['bbox'][1]), 
                    (face['bbox'][2] - face['bbox'][0])) / 2
            
            self.tracks.append({
                'id': len(self.tracks),
                'frames': [frame_idx],
                'bboxes': [face['bbox']],
                'x': [x],
                'y': [y],
                's': [s]
            })
        
        # Remove old tracks
        max_frame = frame_idx - 15  # 15 frames retention
        self.tracks = [t for t in self.tracks if t['frames'][-1] > max_frame]
    
    def _crop_face(self, frame, track_idx):
        """Crop a face from a frame using the latest track information"""
        if track_idx >= len(self.tracks):
            return None
            
        track = self.tracks[track_idx]
        
        if not track['frames']:
            return None
        
        # Use the most recent detection
        bs = track['s'][-1]  # Box size
        cs = self.crop_scale
        bsi = int(bs * (1 + 2 * cs))  # Padding
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Calculate crop coordinates
        y_center = int(track['y'][-1])
        x_center = int(track['x'][-1])
        
        # Calculate boundaries ensuring they're within the image
        y1 = max(0, y_center - int(bs))
        y2 = min(h, y_center + int(bs))
        x1 = max(0, x_center - int(bs))
        x2 = min(w, x_center + int(bs))
        
        # Safety check for valid crop dimensions
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Crop
        face = frame[y1:y2, x1:x2]
        
        # Resize to 224x224 with proper error handling
        try:
            face_resized = cv2.resize(face, (224, 224))
            return face_resized
        except Exception as e:
            print(f"Error resizing face: {e}, dims: {face.shape}")
            return None
    
    def _demo_score_prediction(self, track_id, frame_idx):
        """Generate scores based on simulated audio-visual input, with stable states"""
        # Initialize with default state
        if track_id not in self.demo_speaking_states:
            self.demo_speaking_states[track_id] = {
                'state': 'looking',  # Start with looking by default
                'base_score': 0.6,   # Default looking value
                'last_frame': frame_idx,
                'next_change': frame_idx + 30,
                'manual_override': False,  # Flag to indicate if manually controlled
                'last_random': np.random.random()  # For simulating natural movement
            }
            
        state = self.demo_speaking_states[track_id]
        
        # Skip calculations if manually overridden
        if state.get('manual_override', False):
            return state['base_score']
            
        # Update counters for next calculation
        state['last_frame'] = frame_idx
        
        # Simulate visual attention and voice activity with some randomness
        # This simulates the calculations that would come from real audio/video
        if frame_idx >= state['next_change']:
            # Calculate random values for audio and visual attention
            audio_signal = np.random.random()  # Simulated audio energy
            visual_attention = np.random.random()  # Simulated gaze direction
            
            # Determine state based on audio and visual cues
            if audio_signal > 0.7 and visual_attention > 0.5:
                # High audio energy and looking at camera = speaking
                new_score = 2.6
                next_duration = np.random.randint(20, 40)  # Speaking durations
            elif visual_attention > 0.6:
                # High visual attention but not much audio = looking
                new_score = 0.6
                next_duration = np.random.randint(30, 60)  # Looking durations
            else:
                # Low visual attention = not looking
                new_score = -0.6
                next_duration = np.random.randint(15, 30)  # Not looking durations
                
            # Set the next change time
            state['next_change'] = frame_idx + next_duration
            state['base_score'] = new_score
            
        # Return the current state value (one of the three fixed values)
        return state['base_score']
    
    def _model_score_prediction(self, out):
        """Score prediction using the actual model output - converted to one of three fixed states"""
        try:
            # 1. Direct approach: use class 1 (speaking) raw score
            if out.size(1) > 1:
                raw_score = float(out[0, 1].cpu().detach().numpy())
            else:
                raw_score = float(out[0, 0].cpu().detach().numpy())
                
            # 2. Alternatively, try logit difference
            if out.size(1) > 1:
                logit_diff = float(out[0, 1].cpu().detach().numpy()) - float(out[0, 0].cpu().detach().numpy())
                
                # Use the approach that gives stronger signal
                if abs(logit_diff) > abs(raw_score):
                    score = logit_diff 
                else:
                    score = raw_score
            else:
                score = raw_score
            
            # Scale to reasonable range
            if abs(score) < 0.01:  # If model output is very small
                # Try softmax approach
                if out.size(1) > 1:
                    probs = torch.nn.functional.softmax(out, dim=1)
                    score = float(probs[0, 1].cpu().detach().numpy() - 0.5) * 2
                else:
                    score = float(torch.sigmoid(out).cpu().detach().numpy() - 0.5) * 2
            else:
                # Normalize to [-1, 1] range
                score = min(max(score * 0.5, -1.0), 1.0)
            
            # Map the continuous score to one of three discrete values
            # for more stable display and interpretation
            if score > 0.3:  # Strong positive - speaking
                return 2.6  # Speaking value
            elif score > -0.3:  # Near zero - looking
                return 0.6  # Looking value
            else:  # Negative - not looking
                return -0.6  # Not looking value
                
        except Exception as e:
            print(f"Error in model score prediction: {e}")
            return 0.6  # Default to looking if error
    
    def _evaluate_asd(self, frame_idx):
        """Evaluate ASD for eligible tracks"""
        try:
            # Process each active track
            for i, track in enumerate(self.tracks):
                # Only evaluate tracks with enough history
                if len(track['frames']) < self.min_track_len:
                    continue
                
                # Only process tracks we haven't scored recently
                if i in self.track_scores and frame_idx - self.track_scores[i]['last_eval'] < 3:
                    continue
                
                # Generate a score
                if self.demo_mode:
                    # Use simulated realistic speaking patterns
                    score_value = self._demo_score_prediction(i, frame_idx)
                else:
                    # Use the actual model with current frame and audio
                    
                    # Crop the face from the current frame
                    if not self.frame_buffer:
                        continue
                        
                    face_crop = self._crop_face(self.frame_buffer[-1], i)
                    if face_crop is None:
                        continue
                    
                    # Prepare face data for the model
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    
                    # Get center 112x112 crop
                    h, w = face_gray.shape
                    start_h = max(0, (h - 112) // 2)
                    start_w = max(0, (w - 112) // 2)
                    face_crop_center = face_gray[start_h:start_h+112, start_w:start_w+112]
                    
                    # Check if crop is valid
                    if face_crop_center.shape[0] != 112 or face_crop_center.shape[1] != 112:
                        # Resize to exact dimensions if needed
                        face_crop_center = cv2.resize(face_gray, (112, 112))
                    
                    # Get audio samples
                    with self.buffer_lock:
                        audio_samples = self.audio_buffer.copy()
                        
                    if len(audio_samples) < 400:  # Need at least 400 samples (~25ms)
                        continue
                        
                    # Compute MFCC features
                    mfcc = python_speech_features.mfcc(audio_samples[-2000:], 16000, numcep=13)
                    
                    # Prepare video features
                    video_features = np.array([face_crop_center])
                    
                    # Process with ASD model
                    with torch.no_grad():
                        # Move tensors to the appropriate device
                        device = torch.device(self.device)
                        inputA = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
                        inputV = torch.FloatTensor(video_features).unsqueeze(0).to(device)
                        
                        # Get embeddings from the model
                        embedA = self.asd_model.model.forward_audio_frontend(inputA)
                        embedV = self.asd_model.model.forward_visual_frontend(inputV)
                        
                        # Get final output from audio-visual backend
                        out = self.asd_model.model.forward_audio_visual_backend(embedA, embedV)
                        
                        # Get score from model
                        score_value = self._model_score_prediction(out)
                
                # Store the score
                if i not in self.track_scores:
                    self.track_scores[i] = {'scores': [], 'last_eval': frame_idx}
                    
                self.track_scores[i]['scores'].append(score_value)
                self.track_scores[i]['last_eval'] = frame_idx
                
                # Keep only recent scores
                if len(self.track_scores[i]['scores']) > 5:
                    self.track_scores[i]['scores'] = self.track_scores[i]['scores'][-5:]
                    
                if self.debug_mode:
                    print(f"Track {i} score: {score_value:.2f}")
                    
        except Exception as e:
            print(f"Error in ASD evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_display_frame(self, frame):
        """Create a frame for display with overlaid ASD results"""
        display_frame = frame.copy()
        
        # Add FPS display
        cv2.putText(display_frame, 
                    f"FPS: {self.fps_display:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)
        
        # Add demo mode indicator if active
        if self.demo_mode:
            cv2.putText(display_frame, 
                        "DEMO MODE", 
                        (display_frame.shape[1] - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
        
        # Ensure we show something even when no tracks exist
        if not self.tracks and self.debug_mode:
            cv2.putText(display_frame, 
                        "No face tracks detected", 
                        (50, display_frame.shape[0] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 0, 255), 2)
        
        # Draw all active tracks
        for i, track in enumerate(self.tracks):
            # Display even tracks with minimal history
            if len(track['frames']) == 0:
                continue
            
            # Get the latest detection
            x = track['x'][-1]
            y = track['y'][-1]
            s = track['s'][-1]
            
            # Get the most recent score for this track (no averaging for stability)
            score = 0.0
            if i in self.track_scores and self.track_scores[i]['scores']:
                score = self.track_scores[i]['scores'][-1]  # Use the latest score directly
            
            # Draw box and score
            # Color coding: Green for speaking (2.6), Blue for looking (0.6), Red for not looking (-0.6)
            if abs(score - 2.6) < 0.5:  # Speaking state
                color = (0, 255, 0)  # Green
                status = "SPEAKING"
                thickness = 3  # Thicker for speaking
            elif abs(score - 0.6) < 0.5:  # Looking state
                color = (255, 0, 0)  # Blue (BGR format)
                status = "LOOKING"
                thickness = 2
            else:  # Not looking state
                color = (0, 0, 255)  # Red (BGR format)
                status = "NOT LOOKING"
                thickness = 1  # Thinner for not looking
            
            # Draw bbox - make it prominent
            x1, y1 = int(x-s), int(y-s)
            x2, y2 = int(x+s), int(y+s)
            
            # Make sure coordinates are within frame
            h, w = display_frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            # Draw a more visible box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add score text with background for better visibility
            score_text = f"{score:.1f} - {status}"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_frame, 
                         (x1, y1 - text_size[1] - 5), 
                         (x1 + text_size[0], y1), 
                         color, -1)
            cv2.putText(display_frame, 
                        score_text, 
                        (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 1)
            
        return display_frame
    
    def _process_frames(self):
        """Main processing thread that handles face detection, tracking, and ASD - optimized for performance"""
        frame_idx = 0
        detect_every = 2  # Increased detection frequency to ensure boxes appear
        last_detection = 0
        
        # For performance monitoring
        last_fps_update = time.time()
        frames_processed = 0
        
        while self.running:
            try:
                # Quick check if frames available - non-blocking
                if self.frame_queue.empty():
                    time.sleep(0.001)  # Minimal sleep - just yield CPU
                    continue
                
                # Get frame and start processing
                frame = self.frame_queue.get()
                frames_processed += 1
                
                # Resize frame to smaller size for all processing
                if frame.shape[0] > 480:  # If frame is large, resize for all processing
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                
                # Update internal FPS counter periodically
                current_time = time.time()
                elapsed = current_time - last_fps_update
                if elapsed >= 1.0:
                    self.fps_display = frames_processed / elapsed
                    frames_processed = 0
                    last_fps_update = current_time
                
                # Keep a smaller buffer of frames
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.window_size:
                    self.frame_buffer.pop(0)
                
                # Face detection with more frequent updates to ensure detection boxes appear
                if frame_idx - last_detection >= detect_every:
                    # Always do face detection every few frames to ensure boxes appear
                    faces = self._detect_faces(frame)
                    self.faces_buffer.append(faces)
                    if len(self.faces_buffer) > self.window_size // 2:  # Smaller buffer
                        self.faces_buffer.pop(0)
                    
                    # Update face tracks
                    self._update_tracks(frame_idx, faces)
                    last_detection = frame_idx
                
                # Evaluate ASD more frequently to ensure detection boxes appear
                should_eval_asd = (frame_idx % 2 == 0)  # Every other frame to ensure updates
                
                if should_eval_asd:
                    self._evaluate_asd(frame_idx)
                
                # Always create display frame to ensure detection boxes appear
                display_frame = self._get_display_frame(frame)
                
                # Only keep most recent result to avoid lagging
                while self.result_queue.qsize() > 1:
                    self.result_queue.get_nowait()
                
                self.result_queue.put(display_frame)
                
                frame_idx += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.debug_mode:
                    print(f"Error in processing: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.01)
                continue
    
    def _bb_intersection_over_union(self, boxA, boxB):
        """Calculate IoU between two bounding boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


# Example usage
def main():
    # Try to use CUDA for better performance if available
    try:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("CUDA is available - using GPU for processing")
        else:
            print("CUDA not available - using CPU for processing")
    except:
        use_cuda = False
        print("Error checking CUDA - using CPU for processing")
    
    # Initialize ASD system with appropriate device
    asd = RealtimeASD(use_cuda=use_cuda)
    asd.start()
    
    # Open video and audio capture
    video_capture = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video capture device")
        return
        
    # Try to optimize camera properties for performance
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Small buffer for lower latency
    
    # Try to set codec for better performance
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_capture.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set reasonable window size - no resize needed with smaller window
    window_name = 'Real-time ASD'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)
    
    # Create a black placeholder frame to ensure window is properly initialized
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Initializing...", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, placeholder)
    cv2.waitKey(1)  # This ensures the window gets displayed
    
    print("Starting optimized video capture...")
    print("Controls:")
    print("  'q' - Quit")
    print("  'd' - Toggle debug mode")
    print("  'm' - Toggle demo/model mode")
    print("  's' - Force speaking state in demo mode")
    print("  'l' - Force looking state in demo mode")
    print("  'n' - Force not looking state in demo mode")
    print("  'a' - Return to automatic state changes (cancel manual override)")
    print("  'f' - Toggle display FPS info")
    
    # For performance measurement
    frame_count = 0
    start_time = time.time()
    fps = 0
    skip_count = 0
    last_ui_update = time.time()
    
    # For manual control in demo mode
    manual_state = None
    
    # Additional performance flags
    show_fps = True  # Can be toggled to reduce rendering overhead
    
    try:
        while True:
            # Measure start time for this frame
            frame_start = time.time()
            
            # Capture video frame - optimize by skipping frames if we're falling behind
            if skip_count > 0:
                video_capture.grab()  # Just grab the frame without decoding for better performance
                skip_count -= 1
                continue
                
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame")
                if cv2.waitKey(100) & 0xFF == ord('q'):  # Quick check for exit with shorter wait
                    break
                continue
            
            # Calculate and throttle FPS - only update UI less frequently to save resources
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 0.5:  # Update FPS twice per second
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
                
                # If we're running too slow (below target FPS), start skipping frames
                if fps < 15 and not asd.frame_queue.empty():
                    skip_count = 1  # Skip 1 frame to catch up
            
            # Add frame to processor - avoid copy for better performance
            success = asd.add_frame(frame)  # No need for copy - we're not modifying it
            
            # If the queue is full, skip this frame's audio processing too
            if success:
                # Optimized audio generation - pre-calculated sine wave pattern
                if not hasattr(main, 'speaking_audio'):
                    # Pre-compute the audio patterns (only once)
                    t = np.linspace(0, 0.01, 160)
                    freq = 440  # A4 note
                    main.speaking_audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                    main.looking_audio = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
                    main.noise = np.random.normal(0, 0.05, 160).astype(np.float32)
                    
                # Use pre-computed audio + small noise
                if manual_state == "speaking":
                    dummy_audio = main.speaking_audio + np.random.normal(0, 0.05, 160).astype(np.float32)
                else:
                    dummy_audio = main.looking_audio + np.random.normal(0, 0.05, 160).astype(np.float32)
                    
                asd.add_audio(dummy_audio)
            
            # Handle manual state override for demo mode
            if manual_state and asd.demo_mode:
                # Apply the state change immediately to all tracks
                for track_id in asd.demo_speaking_states:
                    if manual_state == "speaking":
                        asd.demo_speaking_states[track_id]['state'] = 'speaking'
                        asd.demo_speaking_states[track_id]['base_score'] = 2.6
                        asd.demo_speaking_states[track_id]['manual_override'] = True
                    elif manual_state == "looking":
                        asd.demo_speaking_states[track_id]['state'] = 'looking'
                        asd.demo_speaking_states[track_id]['base_score'] = 0.6
                        asd.demo_speaking_states[track_id]['manual_override'] = True
                    elif manual_state == "not_looking":
                        asd.demo_speaking_states[track_id]['state'] = 'not_looking'
                        asd.demo_speaking_states[track_id]['base_score'] = -0.6
                        asd.demo_speaking_states[track_id]['manual_override'] = True
                    elif manual_state == "auto":
                        # Turn off manual override, go back to automatic state changes
                        asd.demo_speaking_states[track_id]['manual_override'] = False
                
                # State has been applied, clear it
                manual_state = None
                last_ui_update = time.time()
            
            # Display results - update every frame for better visibility of detection boxes
            results = asd.get_results()
            if results:
                display_frame = results[-1]
                # Safety check on frame dimensions
                if display_frame.shape[0] > 0 and display_frame.shape[1] > 0:
                    # Add FPS counter if enabled
                    if show_fps:
                        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Add manual control indicator if active
                    if manual_state:
                        cv2.rectangle(display_frame, (20, 20), (620, 60), (0, 0, 255), -1)
                        cv2.putText(display_frame, f"MANUAL: {manual_state.upper()}", 
                                   (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Always display the frame to ensure detection boxes are visible
                    cv2.imshow(window_name, display_frame)
                else:
                    # Fallback to original frame if processed frame is invalid
                    if show_fps:
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow(window_name, frame)
            else:
                # Display original frame with clear indication if no processed results
                cv2.putText(frame, "Detection initializing...", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if show_fps:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, frame)
            
            # Process key presses - non-blocking for better performance
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Toggle debug mode
                asd.debug_mode = not asd.debug_mode
                print(f"Debug mode: {asd.debug_mode}")
            elif key == ord('m'):
                # Toggle demo mode
                asd.demo_mode = not asd.demo_mode
                print(f"Demo mode: {asd.demo_mode}")
            elif key == ord('s'):
                # Force speaking state in demo mode
                manual_state = "speaking"
                print("Manual override: SPEAKING - State will remain until changed")
            elif key == ord('l'):
                # Force looking state in demo mode
                manual_state = "looking"
                print("Manual override: LOOKING - State will remain until changed")
            elif key == ord('n'):
                # Force not looking state in demo mode
                manual_state = "not_looking"
                print("Manual override: NOT LOOKING - State will remain until changed")
            elif key == ord('a'):
                # Return to automatic state changes
                manual_state = "auto"
                print("Returning to automatic state changes based on calculations")
            elif key == ord('f'):
                # Toggle FPS display
                show_fps = not show_fps
                print(f"FPS display: {show_fps}")
            
            # Calculate frame time and potentially sleep to limit CPU usage when running fast
            frame_time = time.time() - frame_start
            if frame_time < 0.01:  # If processing too fast, add a small delay to reduce CPU
                time.sleep(0.005)
                
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
                
    finally:
        # Clean up
        print("Shutting down...")
        asd.stop()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
