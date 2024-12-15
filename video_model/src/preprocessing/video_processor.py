import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms

class VideoFrameProcessor:
    def __init__(self, face_size=(112, 112)):
        self.face_size = face_size
        self.face_detector = MTCNN(
            image_size=face_size[0],
            margin=10,
            keep_all=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face = self.face_detector(frame_rgb)
            return face if face is not None else None
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None