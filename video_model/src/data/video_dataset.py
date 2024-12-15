import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from ..preprocessing.video_processor import VideoFrameProcessor

class GRIDDataset(Dataset):
    def __init__(self, root_dir, subject_list, sequence_length=16, stride=8):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.frame_processor = VideoFrameProcessor()
        
        # Collect all video paths and labels
        self.samples = []
        for subject in subject_list:
            subject_dir = os.path.join(root_dir, f's{subject}')
            for video_file in os.listdir(subject_dir):
                if video_file.endswith('.mpg'):
                    video_path = os.path.join(subject_dir, video_file)
                    # Parse label from filename (you'll need to adapt this based on GRID naming convention)
                    label = self._parse_label(video_file)
                    self.samples.append((video_path, label))

    def _parse_label(self, filename):
        # Remove extension
        filename = os.path.splitext(filename)[0]
        
        # Get the alignment file path (assuming it's in an 'align' subdirectory)
        align_path = os.path.join(
            os.path.dirname(self.root_dir),
            'align',
            filename + '.align'
        )
        
        try:
            # Read alignment file
            with open(align_path, 'r') as f:
                lines = f.readlines()
                
            # Parse alignment times
            # GRID alignment files contain start_time end_time word
            speaking_segments = []
            for line in lines:
                start, end, word = line.strip().split()
                start = int(start)
                end = int(end)
                # Skip silence markers
                if word != 'sil':
                    speaking_segments.append((start, end))
            
            # Calculate which frame this corresponds to
            # GRID videos are 25fps
            frame_number = int(filename[-5])  # Extract frame number from filename
            frame_time = frame_number * (1000 / 25)  # Convert to milliseconds
            
            # Check if this frame falls within any speaking segment
            for start, end in speaking_segments:
                if start <= frame_time <= end:
                    return 1
                    
            return 0
            
        except FileNotFoundError:
            # If alignment file is not found, fall back to basic heuristic
            # Check if it's the start or end of the video (typically silent)
            frame_number = int(filename[-5])
            total_frames = 75  # GRID videos are 3 seconds at 25fps
            if frame_number < 10 or frame_number > total_frames - 10:
                return 0
            return 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_and_process_video(video_path)
        sequences = self._create_sequences(frames)
        return sequences, label

    def _load_and_process_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.frame_processor.process_frame(frame)
            if processed_frame is not None:
                frames.append(processed_frame)
        cap.release()
        return frames

    def _create_sequences(self, frames):
        sequences = []
        for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
            sequence = frames[i:i + self.sequence_length]
            if len(sequence) == self.sequence_length:
                sequences.append(torch.stack(sequence))
        return torch.stack(sequences) if sequences else torch.empty(0)