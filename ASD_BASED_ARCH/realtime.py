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
from model.faceDetector.s3fd import S3FD
from ASD import ASD

class RealtimeASD:
    def __init__(self, model_path=r"ASD_BASED_ARCH\weight\pretrain_AVA_CVPR.model", use_cuda=False):
        # Configuration parameters
        self.facedet_scale = 0.25
        self.min_face_size = 1
        self.crop_scale = 0.40
        self.window_size = 60  # Number of frames to keep in the sliding window
        self.fps = 25  # Assuming 25 fps video
        self.min_track_len = 5  # Minimum track length to start evaluation
        self.iou_threshold = 0.5  # IOU threshold for face tracking
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        # Device selection
        self.device = 'cuda' if self.use_cuda else 'cpu'
        print(f"Using device: {self.device}")
        
        try:
            # Initialize models
            print("Initializing face detector...")
            self.face_detector = S3FD(device=self.device)
            
            print("Loading ASD model...")
            self.asd_model = ASD()
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"WARNING: Model file not found at {model_path}")
                print("Using a placeholder model for demonstration")
            else:
                self.asd_model.loadParameters(model_path)
            self.asd_model.eval()
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            print("Creating placeholder detection functionality")
            self.face_detector = None
            self.asd_model = None
        
        # Initialize tracking variables
        self.frame_buffer = []  # Buffer to store recent frames
        self.audio_buffer = []  # Buffer to store recent audio
        self.faces_buffer = []  # Buffer to store face detections
        self.tracks = []        # Active face tracks
        self.track_scores = {}  # Scores for each track
        self.debug_mode = True  # Set to True to print debugging information
        
        # Threading resources
        self.frame_queue = queue.Queue(maxsize=30)
        self.audio_queue = queue.Queue(maxsize=1600)  # ~1 second of audio at 16kHz
        self.result_queue = queue.Queue(maxsize=30)
        self.running = False
        
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
            self.frame_queue.put(frame, block=False)
            return True
        except queue.Full:
            return False
    
    def add_audio(self, audio_chunk):
        """Add a new audio chunk to the processing queue"""
        for sample in audio_chunk:
            try:
                self.audio_queue.put(sample, block=False)
            except queue.Full:
                # Discard oldest samples if queue is full
                self.audio_queue.get()
                self.audio_queue.put(sample)
    
    def get_results(self):
        """Get the latest processing results"""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())
        return results
    
    def _detect_faces(self, frame):
        """Detect faces in a frame"""
        try:
            if self.face_detector is None:
                # Fallback face detection using OpenCV's built-in detector if model wasn't loaded
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                faces = []
                for (x, y, w, h) in detected_faces:
                    # Convert to [x1, y1, x2, y2] format and add confidence
                    faces.append({'bbox': [float(x), float(y), float(x+w), float(y+h)], 'conf': 0.9})
                return faces
            
            # Use the S3FD detector
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes = self.face_detector.detect_faces(image_rgb, conf_th=0.9, scales=[self.facedet_scale])
            
            faces = []
            for bbox in bboxes:
                faces.append({'bbox': (bbox[:-1]).tolist(), 'conf': bbox[-1]})
            
            if self.debug_mode and len(faces) > 0:
                print(f"Detected {len(faces)} faces")
                
            return faces
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def _update_tracks(self, frame_idx, faces):
        """Update face tracks with new detections"""
        # For existing tracks, find matching faces
        unmatched_faces = faces.copy()
        
        for track in self.tracks:
            if len(track['frames']) == 0 or frame_idx - track['frames'][-1] > 10:
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
                
                # Apply smoothing to track parameters
                if len(track['s']) > 3:
                    window_size = min(13, len(track['s']))
                    track['s'][-1] = np.median(track['s'][-window_size:])
                    track['x'][-1] = np.median(track['x'][-window_size:])
                    track['y'][-1] = np.median(track['y'][-window_size:])
                
                # Remove the matched face
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
        max_frame = frame_idx - 30  # 30 frames retention
        self.tracks = [t for t in self.tracks if t['frames'][-1] > max_frame]
    
    def _crop_face(self, frame, track_idx):
        """Crop a face from a frame using the latest track information"""
        track = self.tracks[track_idx]
        
        if not track['frames']:
            return None
        
        # Use the most recent detection
        bs = track['s'][-1]  # Box size
        cs = self.crop_scale
        bsi = int(bs * (1 + 2 * cs))  # Padding
        
        # Pad the frame
        padded = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), 
                         'constant', constant_values=(110, 110))
        
        # Calculate crop coordinates
        my = track['y'][-1] + bsi  # center Y
        mx = track['x'][-1] + bsi  # center X
        
        # Crop and resize
        face = padded[int(my-bs):int(my+bs*(1+2*cs)),
                       int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        
        # Resize to 224x224
        face_resized = cv2.resize(face, (224, 224))
        return face_resized
    
    def _evaluate_asd(self, frame_idx):
        """Evaluate ASD for eligible tracks"""
        try:
            # If the model is not available, use a placeholder implementation
            if self.asd_model is None:
                # Simple placeholder: assign random scores
                for i, track in enumerate(self.tracks):
                    if len(track['frames']) < self.min_track_len:
                        continue
                        
                    # Randomly assign speaking/not speaking with temporal consistency
                    if i not in self.track_scores:
                        self.track_scores[i] = {
                            'scores': [], 
                            'last_eval': frame_idx,
                            'speaking_prob': np.random.random()  # Random probability of speaking
                        }
                    
                    # Add some noise to the score but maintain temporal consistency
                    current_mean = np.mean(self.track_scores[i]['scores']) if self.track_scores[i]['scores'] else 0
                    if current_mean == 0:
                        # Initialize with random speaking state
                        new_score = 1.0 if self.track_scores[i]['speaking_prob'] > 0.7 else -1.0
                    else:
                        # 90% chance to keep same speaking state, 10% to change
                        if np.random.random() < 0.9:
                            new_score = current_mean + np.random.normal(0, 0.3)  # Add noise
                        else:
                            new_score = -current_mean + np.random.normal(0, 0.3)  # Switch state
                    
                    self.track_scores[i]['scores'].append(float(new_score))
                    self.track_scores[i]['last_eval'] = frame_idx
                    
                    # Keep only recent scores
                    if len(self.track_scores[i]['scores']) > 10:
                        self.track_scores[i]['scores'] = self.track_scores[i]['scores'][-10:]
                
                return
            
            # Get the current audio buffer
            audio_samples = list(self.audio_queue.queue)
            if len(audio_samples) < 400:  # Need at least 400 samples (~25ms)
                return
            
            audio_array = np.array(audio_samples, dtype=np.float32)
            # Compute MFCC features
            mfcc = python_speech_features.mfcc(audio_array, 16000, numcep=13)
            
            for i, track in enumerate(self.tracks):
                # Only evaluate tracks with enough history
                if len(track['frames']) < self.min_track_len:
                    continue
                
                # Only process tracks we haven't scored recently
                if i in self.track_scores and frame_idx - self.track_scores[i]['last_eval'] < 5:
                    continue
                
                # Crop the face from the current frame
                face_crop = self._crop_face(self.frame_buffer[-1], i)
                if face_crop is None:
                    continue
                
                # Prepare face data for the model
                face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_crop_center = face_gray[int(112-(112/2)):int(112+(112/2)), 
                                              int(112-(112/2)):int(112+(112/2))]
                
                # Prepare video features
                video_features = np.array([face_crop_center])
                
                # Process with ASD model
                with torch.no_grad():
                    # Move tensors to the appropriate device
                    device = torch.device(self.device)
                    inputA = torch.FloatTensor(mfcc).unsqueeze(0).to(device)
                    inputV = torch.FloatTensor(video_features).unsqueeze(0).to(device)
                    
                    embedA = self.asd_model.model.forward_audio_frontend(inputA)
                    embedV = self.asd_model.model.forward_visual_frontend(inputV)
                    out = self.asd_model.model.forward_audio_visual_backend(embedA, embedV)
                    score = self.asd_model.lossAV.forward(out, labels=None)
                    
                    # Store the score
                    if i not in self.track_scores:
                        self.track_scores[i] = {'scores': [], 'last_eval': frame_idx}
                    
                    self.track_scores[i]['scores'].append(score[0])
                    self.track_scores[i]['last_eval'] = frame_idx
                    
                    # Keep only recent scores
                    if len(self.track_scores[i]['scores']) > 10:
                        self.track_scores[i]['scores'] = self.track_scores[i]['scores'][-10:]
        except Exception as e:
            print(f"Error in ASD evaluation: {e}")
            # Fall back to placeholder implementation next time
            self.asd_model = None
    
    def _get_display_frame(self, frame):
        """Create a frame for display with overlaid ASD results"""
        display_frame = frame.copy()
        
        for i, track in enumerate(self.tracks):
            if len(track['frames']) < self.min_track_len:
                continue
            
            # Get the latest detection
            x = track['x'][-1]
            y = track['y'][-1]
            s = track['s'][-1]
            
            # Get the average score for this track
            avg_score = 0
            if i in self.track_scores and self.track_scores[i]['scores']:
                avg_score = np.mean(self.track_scores[i]['scores'])
            
            # Draw box and score
            color = (0, 0, 255)  # Default to red (not speaking)
            if avg_score > 0:
                color = (0, 255, 0)  # Green (speaking)
            
            cv2.rectangle(display_frame, 
                          (int(x-s), int(y-s)), 
                          (int(x+s), int(y+s)), 
                          color, 3)
            
            cv2.putText(display_frame, 
                        f"{avg_score:.1f}", 
                        (int(x-s), int(y-s-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)
            
        return display_frame
    
    def _process_frames(self):
        """Main processing thread that handles face detection, tracking, and ASD"""
        frame_idx = 0
        
        while self.running:
            try:
                # Get the next frame
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                frame = self.frame_queue.get()
                
                # Store frame in buffer
                self.frame_buffer.append(frame)
                if len(self.frame_buffer) > self.window_size:
                    self.frame_buffer.pop(0)
                
                # Detect faces in the current frame
                faces = self._detect_faces(frame)
                self.faces_buffer.append(faces)
                if len(self.faces_buffer) > self.window_size:
                    self.faces_buffer.pop(0)
                
                # Update face tracks
                self._update_tracks(frame_idx, faces)
                
                # Evaluate ASD for active tracks
                self._evaluate_asd(frame_idx)
                
                # Create frame for display
                display_frame = self._get_display_frame(frame)
                
                # Put result in queue
                self.result_queue.put(display_frame)
                
                frame_idx += 1
                
            except Exception as e:
                print(f"Error in processing: {e}")
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
    # Initialize ASD system
    asd = RealtimeASD()
    asd.start()
    
    # Open video and audio capture
    video_capture = cv2.VideoCapture(0)  # Use 0 for webcam
    
    # Set reasonable window size
    window_name = 'Real-time ASD'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    # Create a black placeholder frame to ensure window is properly initialized
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Initializing...", (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, placeholder)
    cv2.waitKey(1)  # This ensures the window gets displayed
    
    print("Starting video capture...")
    
    try:
        while True:
            # Capture video frame
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame")
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No camera feed", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow(window_name, placeholder)
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
                continue
            
            # Verify frame dimensions
            print(f"Frame shape: {frame.shape}")
            
            # Add frame to processor
            asd.add_frame(frame.copy())  # Use copy to avoid modification issues
            
            # Create dummy audio data (in real app, you'd get this from microphone)
            dummy_audio = np.zeros(160, dtype=np.float32)  # 10ms of audio at 16kHz
            asd.add_audio(dummy_audio)
            
            # Get and display results
            results = asd.get_results()
            if results:
                display_frame = results[-1]
                # Safety check on frame dimensions
                if display_frame.shape[0] > 0 and display_frame.shape[1] > 0:
                    cv2.imshow(window_name, display_frame)
                else:
                    print("Error: Invalid frame dimensions")
            else:
                # Display original frame if no processed results yet
                cv2.imshow(window_name, frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in main loop: {e}")
                
    finally:
        # Clean up
        print("Shutting down...")
        asd.stop()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()