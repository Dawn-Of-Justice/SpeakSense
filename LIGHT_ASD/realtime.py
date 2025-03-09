# Modified realtime3.py for faster ASD processing

import threading
import pyaudio
import cv2
import os
import numpy as np
import wave
from queue import Queue
import time
import python_speech_features
import torch
import math
# To run seperatley uncomment the below import Salo and Rahul
# from model.faceDetector.s3fd import S3FD
# from ASD import ASD
 
# And comment this
from .model.faceDetector.s3fd import S3FD
from .ASD import ASD

shared_state = None

# Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ASD Model
asd_model = ASD(device=device)
model_path = "LIGHT_ASD/weight/finetuning_TalkSet.model"
asd_model.loadParameters(model_path)
asd_model.eval()  # Set model to evaluation mode
print(f"Model loaded from {model_path}")

# Create queues for audio data and face frames
audio_queue = Queue()
face_frames_queue = Queue()

# Minimum sizes required by the model
MIN_AUDIO_FRAMES = 100  # 100 frames at 100fps = 1 second
MIN_VIDEO_FRAMES = 25   # 25 frames at 25fps = 1 second

def extract_MFCC(audio_data, sr=16000):
    """Extract MFCC features from audio data"""
    mfcc = python_speech_features.mfcc(audio_data, sr, numcep=13, winlen=0.025, winstep=0.010)
    return mfcc

def getfaces(frame, idx, det=None):
    """Detect faces in the frame with optional cached detector"""
    if det is None:
        det = S3FD(device='cpu')
        
    dets = []
    image = frame
    imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = det.detect_faces(imageNumpy, conf_th=0.9, scales=[0.25])  # Use smaller scale for speed
    dets.append([])
    for bbox in bboxes:
        dets[-1].append({'frame':idx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
    
    return dets, det

def evaluate_speaker(audio_data, face_frames):
    """Process audio and face frames to detect active speaker with zero-padding for smaller inputs"""
    # Process frames as before
    processed_frames = []
    for frame in face_frames:
        if frame is None or frame.size == 0:
            continue
        
        # Convert to grayscale if necessary
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        # Center crop
        h, w = gray_frame.shape
        center_h, center_w = h // 2, w // 2
        crop_size = 112
        crop = gray_frame[
            max(0, center_h - crop_size // 2):min(h, center_h + crop_size // 2),
            max(0, center_w - crop_size // 2):min(w, center_w + crop_size // 2)
        ]
        
        # Ensure the crop is exactly the right size by padding if needed
        if crop.shape != (crop_size, crop_size):
            padded_crop = np.zeros((crop_size, crop_size), dtype=crop.dtype)
            h_crop, w_crop = crop.shape
            h_start = (crop_size - h_crop) // 2
            w_start = (crop_size - w_crop) // 2
            padded_crop[h_start:h_start+h_crop, w_start:w_start+w_crop] = crop
            crop = padded_crop
            
        if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
            processed_frames.append(crop)
    
    if len(processed_frames) == 0:
        return 0.0
    
    # Extract audio features
    audio_feature = extract_MFCC(audio_data)
    
    # Convert to numpy arrays
    video_feature = np.array(processed_frames)
    
    # Zero-pad if not enough data instead of skipping
    if audio_feature.shape[0] < MIN_AUDIO_FRAMES:
        pad_length = MIN_AUDIO_FRAMES - audio_feature.shape[0]
        audio_feature = np.pad(audio_feature, ((0, pad_length), (0, 0)), 'constant')
    
    if video_feature.shape[0] < MIN_VIDEO_FRAMES:
        # Handle case where we don't have enough video frames
        if video_feature.shape[0] == 0:
            # If no frames, create empty ones
            video_feature = np.zeros((MIN_VIDEO_FRAMES, crop_size, crop_size), dtype=np.float32)
        else:
            # Duplicate the last frame to reach the minimum
            last_frame = video_feature[-1]
            num_to_add = MIN_VIDEO_FRAMES - video_feature.shape[0]
            padding_frames = np.tile(last_frame, (num_to_add, 1, 1))
            video_feature = np.vstack((video_feature, padding_frames))
    
    # Ensure audio frames are multiples of 4 for the model
    audio_frames = audio_feature.shape[0]
    if audio_frames % 4 != 0:
        audio_frames = audio_frames - (audio_frames % 4)
        audio_feature = audio_feature[:audio_frames, :]
    
    # Trim video to match the model's expectations if needed
    video_feature = video_feature[:MIN_VIDEO_FRAMES, :, :]
    
    # Process with the model
    with torch.no_grad():
        # Use batch processing to speed up inference
        inputA = torch.FloatTensor(audio_feature).unsqueeze(0).to(device)
        inputV = torch.FloatTensor(video_feature).unsqueeze(0).to(device)
        
        # Get embeddings
        embedA = asd_model.model.forward_audio_frontend(inputA)
        embedV = asd_model.model.forward_visual_frontend(inputV)
        
        # Ensure embeddings have matching time dimension
        if embedA.shape[1] > embedV.shape[1]:
            embedA = embedA[:, :embedV.shape[1], :]
        elif embedV.shape[1] > embedA.shape[1]:
            embedV = embedV[:, :embedA.shape[1], :]
        
        # Get final output
        out = asd_model.model.forward_audio_visual_backend(embedA, embedV)
        
        # Get score
        pred_score = asd_model.lossAV.forward(out, labels=None)
        speaking_score = pred_score[-1].item()  # Probability of speaking class
        
    return speaking_score

def crop_face(frame, det, cs=0.40):
    """Crop a face from a frame using detection coordinates"""
    # Extract bbox coordinates
    bbox = det['bbox']
    x1, y1, x2, y2 = bbox
    
    # Calculate center and size
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    size = max((y2 - y1), (x2 - x1)) / 2
    
    # Calculate padding
    bsi = int(size * (1 + 2 * cs))
    
    # Fast cropping without unnecessary padding
    h, w = frame.shape[:2]
    crop_y1 = max(0, int(center_y - size * (1 + cs)))
    crop_y2 = min(h, int(center_y + size * (1 + cs)))
    crop_x1 = max(0, int(center_x - size * (1 + cs)))
    crop_x2 = min(w, int(center_x + size * (1 + cs)))
    
    # Crop and resize face
    face = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Only resize if we have a valid face crop
    if face.size > 0:
        face = cv2.resize(face, (224, 224))
    
    return face

def record_audio(duration=0.5, sample_rate=16000):
    """Record audio continuously in the background with shorter duration chunks"""
    global audio_queue
    
    # Keep recording in a loop
    while True:
        chunk = 1024
        audio_format = pyaudio.paInt16
        channels = 1
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
        
        frames = []
        
        for i in range(0, int(sample_rate / chunk * duration)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        # Put the audio data in the queue
        audio_queue.put(audio_data)

def main(run_sub_audio_thread=True):
    global shared_state
    
    # Initialize camera with lower resolution for speed
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    count = 0
    
    # Initialize face detector once to reuse
    face_detector = S3FD(device='cpu')
    
    # Buffer min size reduced for faster updates
    max_buffer_size = 15  # Reduced from 25
    min_buffer_size = 10  # Minimum size required for evaluation
    
    if run_sub_audio_thread:
        # Start audio recording thread with shorter chunks
        audio_thread = threading.Thread(target=record_audio, args=(0.5, 16000))
        audio_thread.daemon = True
        audio_thread.start()
    
    # For tracking face identities
    tracked_faces = {}
    next_face_id = 0
    
    # Last processing timestamp
    last_processed = time.time()
    process_interval = 0.1  # Process every 100ms instead of waiting for full buffers
    
    # Main loop
    while True:
        ret, frame = cam.read()
        
        if not ret:
            break
        
        # Skip frames to increase speed
        if count % 2 != 0:  # Process every other frame
            count += 1
            continue
            
        # Reduce resolution for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect faces and reuse detector
        dets, face_detector = getfaces(frame, count, face_detector)
        
        # List to store current face detections with IDs
        current_faces = []
        
        # Process each detected face
        for i in range(len(dets)):
            # Check if there are any detections in this group
            if dets[i] and len(dets[i]) > 0:
                for face_det in dets[i]:
                    # Get face coordinates
                    bbox = face_det['bbox']
                    face_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    
                    # Crop the face
                    cropped_face = crop_face(frame, face_det)
                    
                    # Skip if invalid crop
                    if cropped_face.size == 0:
                        continue
                    
                    # Find closest tracked face
                    face_id = next_face_id
                    min_dist = float('inf')
                    for fid, face_info in tracked_faces.items():
                        last_center = face_info['center']
                        dist = ((face_center[0] - last_center[0])**2 + 
                                (face_center[1] - last_center[1])**2)**0.5
                        if dist < min_dist and dist < 100:  # Threshold for same identity
                            min_dist = dist
                            face_id = fid
                    
                    # Update or create tracked face
                    if face_id == next_face_id:
                        tracked_faces[face_id] = {
                            'buffer': [cropped_face],
                            'center': face_center,
                            'speaking_score': 0.0,
                            'last_update': time.time()
                        }
                        next_face_id += 1
                    else:
                        # Add to buffer and update center
                        tracked_faces[face_id]['buffer'].append(cropped_face)
                        tracked_faces[face_id]['center'] = face_center
                        tracked_faces[face_id]['last_update'] = time.time()
                        
                        # Limit buffer size
                        if len(tracked_faces[face_id]['buffer']) > max_buffer_size:
                            tracked_faces[face_id]['buffer'].pop(0)
                    
                    current_faces.append(face_id)
                    
                    # Draw rectangle and ID
                    color = (0, 0, 255)  # Default color (red)
                    shared_state = False
                    if tracked_faces[face_id]['speaking_score'] > 0.5:
                        color = (0, 255, 0)  # Green for speaking
                        shared_state = True
                        
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), color, 2)  # Thinner lines for speed
                    
                    # Display speaking score with simplified text
                    score_text = f"{face_id}:{tracked_faces[face_id]['speaking_score']:.1f}"
                    cv2.putText(frame, score_text, 
                                (int(bbox[0]), int(bbox[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)  # Smaller text
        
        # Process audio and update speaking scores at regular intervals
        current_time = time.time()
        if current_time - last_processed > process_interval and not audio_queue.empty():
            audio_data = audio_queue.get()
            
            # Process faces with sufficient buffer size
            for face_id, face_info in tracked_faces.items():
                if len(face_info['buffer']) >= min_buffer_size:
                    # Run ASD evaluation with the current buffer, even if small
                    speaking_score = evaluate_speaker(audio_data, face_info['buffer'])
                    # Apply smoothing to avoid rapid changes
                    if 'speaking_score' in face_info:
                        face_info['speaking_score'] = 0.7 * face_info['speaking_score'] + 0.3 * speaking_score
                    else:
                        face_info['speaking_score'] = speaking_score
            
            last_processed = current_time
        
        # Remove faces that haven't been seen recently (shorter timeout)
        faces_to_remove = []
        for face_id, face_info in tracked_faces.items():
            if current_time - face_info['last_update'] > 2.0:  # Reduced from 5.0
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            del tracked_faces[face_id]
        
        # Display frame (optional - can be disabled for headless operation)
        cv2.imshow("Active Speaker Detection", frame)
        count += 1
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Cleanup
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()