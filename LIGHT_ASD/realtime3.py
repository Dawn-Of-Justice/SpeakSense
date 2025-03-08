from model.faceDetector.s3fd import S3FD
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
from ASD import ASD
import math

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

def extract_MFCC(audio_data, sr=16000):
    """Extract MFCC features from audio data"""
    mfcc = python_speech_features.mfcc(audio_data, sr, numcep=13, winlen=0.025, winstep=0.010)
    return mfcc

def getfaces(frame, idx):
    """Detect faces in the frame"""
    dets = []
    DET = S3FD(device='cpu')
    image = frame
    imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[0.25])
    dets.append([])
    for bbox in bboxes:
        dets[-1].append({'frame':idx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
    
    return dets

# Modify the evaluate_speaker function to ensure embeddings have matching dimensions
def evaluate_speaker(audio_data, face_frames):
    """
    Process audio and face frames to detect active speaker
    """
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
            center_h - crop_size // 2:center_h + crop_size // 2,
            center_w - crop_size // 2:center_w + crop_size // 2
        ]
        
        if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
            processed_frames.append(crop)
    
    if len(processed_frames) == 0:
        return 0.0
    
    # Extract audio features
    audio_feature = extract_MFCC(audio_data)
    
    # Convert to numpy arrays
    video_feature = np.array(processed_frames)
    
    # Skip if not enough data
    if audio_feature.shape[0] < 100 or video_feature.shape[0] < 25:
        return 0.0
    
    # Calculate a common timeframe for both modalities
    # Audio: 100 frames per second
    # Video: 25 frames per second
    # We need to make these compatible for the backend
    audio_seconds = audio_feature.shape[0] / 100.0
    video_seconds = video_feature.shape[0] / 25.0
    
    # Use the shorter duration (usually video will be shorter)
    common_seconds = min(audio_seconds, video_seconds)
    
    # Trim to appropriate length
    audio_frames = int(common_seconds * 100)
    video_frames = int(common_seconds * 25)
    
    # Make sure we have multiples of 4 for audio as required
    audio_frames = audio_frames - (audio_frames % 4)
    
    # Trim both features to match the common timeframe
    audio_feature = audio_feature[:audio_frames, :]
    video_feature = video_feature[:video_frames, :, :]
    
    # Process with the model
    with torch.no_grad():
        inputA = torch.FloatTensor(audio_feature).unsqueeze(0).to(device)
        inputV = torch.FloatTensor(video_feature).unsqueeze(0).to(device)
        
        # Get embeddings
        embedA = asd_model.model.forward_audio_frontend(inputA)
        embedV = asd_model.model.forward_visual_frontend(inputV)
        
        # Debug the shapes
        # print(f"Audio embedding shape: {embedA.shape}")
        # print(f"Video embedding shape: {embedV.shape}")
        
        # Ensure embeddings have matching time dimension
        if embedA.shape[1] > embedV.shape[1]:
            embedA = embedA[:, :embedV.shape[1], :]
        elif embedV.shape[1] > embedA.shape[1]:
            embedV = embedV[:, :embedA.shape[1], :]
            
        # Verify shapes match after adjustment
        # print(f"Adjusted audio shape: {embedA.shape}")
        # print(f"Adjusted video shape: {embedV.shape}")
        
        # Get final output
        out = asd_model.model.forward_audio_visual_backend(embedA, embedV)
        
        # Get score
        pred_score = asd_model.lossAV.forward(out, labels=None)
        # print(pred_score)
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
    
    # Pad the image
    padded_frame = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
    
    # Adjust coordinates for padded frame
    center_y += bsi
    center_x += bsi
    
    # Calculate crop coordinates
    crop_y1 = int(center_y - size)
    crop_y2 = int(center_y + size * (1 + 2 * cs))
    crop_x1 = int(center_x - size * (1 + cs))
    crop_x2 = int(center_x + size * (1 + cs))
    
    # Ensure coordinates are within bounds
    h, w = padded_frame.shape[:2]
    crop_y1 = max(0, crop_y1)
    crop_y2 = min(h, crop_y2)
    crop_x1 = max(0, crop_x1)
    crop_x2 = min(w, crop_x2)
    
    # Crop and resize face
    face = padded_frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Only resize if we have a valid face crop
    if face.size > 0:
        face = cv2.resize(face, (224, 224))
    
    return face

def record_audio(duration=2, sample_rate=16000):
    """Record audio continuously in the background"""
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
        
        # print("Recording audio...")
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
        
        print(f"Audio recorded: {len(audio_data)} samples")

if __name__ == "__main__":
    # Initialize camera
    cam = cv2.VideoCapture(0)
    count = 0
    
    # Face buffer to store a sequence of face frames
    face_buffer = []
    max_buffer_size = 25  # 25 frames = 1 second at 25 fps
    
    # Start audio recording thread
    audio_thread = threading.Thread(target=record_audio, args=(2, 16000))
    audio_thread.daemon = True
    audio_thread.start()
    
    # Create directory for saving audio files if needed
    os.makedirs("audio_samples", exist_ok=True)
    
    # For tracking face identities (very basic implementation)
    tracked_faces = {}
    next_face_id = 0
    
    # Main loop
    while True:
        ret, frame = cam.read()
        
        if not ret:
            break
        
        # Detect faces
        dets = getfaces(frame, count)
        
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
                    
                    # Find closest tracked face (simple tracking by center position)
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
                    if tracked_faces[face_id]['speaking_score'] > 0.5:
                        color = (0, 255, 0)  # Green for speaking
                        
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), color, 4)
                    
                    # Display speaking score
                    score_text = f"ID: {face_id}, Score: {tracked_faces[face_id]['speaking_score']:.2f}"
                    cv2.putText(frame, score_text, 
                                (int(bbox[0]), int(bbox[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Remove faces that haven't been seen recently
        current_time = time.time()
        faces_to_remove = []
        for face_id, face_info in tracked_faces.items():
            if current_time - face_info['last_update'] > 5.0:
                faces_to_remove.append(face_id)
        
        for face_id in faces_to_remove:
            del tracked_faces[face_id]
        
        # Get latest audio if available
        if not audio_queue.empty():
            audio_data = audio_queue.get()
            
            # Process each tracked face with the audio
            for face_id, face_info in tracked_faces.items():
                if len(face_info['buffer']) >= 15:  # Need at least ~0.6s of video
                    # Run ASD evaluation
                    speaking_score = evaluate_speaker(audio_data, face_info['buffer'])
                    tracked_faces[face_id]['speaking_score'] = speaking_score
        
        # Display frame
        cv2.imshow("Active Speaker Detection", frame)
        count += 1
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Cleanup
    cam.release()
    cv2.destroyAllWindows()