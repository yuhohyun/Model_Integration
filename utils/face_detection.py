# face_detection.py
import cv2
from facenet_pytorch import MTCNN
import torch
from PIL import Image

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceDetector:
    def __init__(self, device):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, device=device)
    
    def detect_and_crop(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(Image.fromarray(img_rgb))
        
        cropped_faces = []
        
        if boxes is not None:
            for box in boxes:
                xmin, ymin, xmax, ymax = [int(b) for b in box]
                face = img[ymin:ymax, xmin:xmax]
                face_resized = cv2.resize(face, (224, 224))
                cropped_faces.append(face_resized)
                
        return cropped_faces
