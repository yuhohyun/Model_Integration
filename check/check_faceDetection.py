import sys
import os

# 스크립트가 위치한 디렉토리의 상위 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.face_detection import FaceDetector
import cv2
import torch

# 비디오 파일 경로
video_path = 'C:/Project/Model_Integration/images/free_video1.mp4'

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 얼굴 검출 객체 초기화
detector = FaceDetector(device)

# 비디오 캡처 객체 초기화
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    ret, frame = cap.read()
    if ret:
        # 얼굴 검출 및 크롭
        cropped_faces = detector.detect_and_crop(frame)
        
        # 검출된 얼굴 이미지 표시
        for i, face in enumerate(cropped_faces):
            cv2.imshow(f'Face {i+1}', face)
            
        cv2.waitKey(0)  # 키 입력이 있을 때까지 대기
        cv2.destroyAllWindows()
    cap.release()
