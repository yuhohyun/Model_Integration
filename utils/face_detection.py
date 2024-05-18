from facenet_pytorch import MTCNN
import torch
from PIL import Image
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, device):
        self.device = device
        self.mtcnn = MTCNN(keep_all=True, landmarks=True, device=device)
        self.prev_landmarks = None

    def detect_and_crop(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Detect faces and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(Image.fromarray(img_rgb), landmarks=True)
        
        cropped_faces = []
        expression_change_detected = False
        
        if boxes is not None:
            for box, landmark in zip(boxes, landmarks):
                if landmark is not None and self.prev_landmarks is not None:
                    # Calculate expression change based on eye landmarks
                    expression_change_detected = self.detect_expression_change(self.prev_landmarks, landmark)

                xmin, ymin, xmax, ymax = [int(b) for b in box]
                face = img[ymin:ymax, xmin:xmax]
                face_resized = cv2.resize(face, (224, 224))
                cropped_faces.append(face_resized)
                
                # Update previous landmarks if expression change is detected
                if expression_change_detected:
                    self.prev_landmarks = np.array(landmark)

        return cropped_faces, expression_change_detected

    def detect_expression_change(self, prev_landmarks, current_landmarks):
        """Detect expression change based on eye landmarks."""
        eye_indices = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]  # Indices for eye landmarks
        prev_eye_points = np.array([prev_landmarks[i] for i in eye_indices])
        current_eye_points = np.array([current_landmarks[i] for i in eye_indices])
        
        # Calculate Euclidean distances for eye landmarks between frames
        distances = np.linalg.norm(prev_eye_points - current_eye_points, axis=1)
        
        # Check if the change is significant
        return np.any(distances > 5)  # Threshold is set arbitrarily
