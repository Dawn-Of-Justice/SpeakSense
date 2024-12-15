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
        # Implement based on GRID filename convention
        # Example: if speaking status is in filename
        return 1 if "speaking" in filename else 0

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