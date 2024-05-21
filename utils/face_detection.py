from moviepy.editor import VideoFileClip
import cv2
import numpy as np
import os
import dlib
import shutil

class ProcessVideo:
    def __init__(self, input_video_path, output_audio_path, output_video_path, output_folder, frame_interval=5, margin=20):
        # 경로 초기화
        self.input_video_path = input_video_path
        self.output_audio_path = output_audio_path
        self.output_video_path = output_video_path
        self.output_folder = output_folder
        self.temp_folder = os.path.join(output_folder, 'temp')
        self.margin = margin  # 얼굴 주위의 배경을 포함하는 마진 설정
        # 비디오 파일명을 디렉토리 이름으로 사용
        video_name = os.path.splitext(os.path.basename(input_video_path))[0]
        self.face_folder = os.path.join(output_folder, video_name)
        # 출력 폴더가 없으면 생성
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)
        if not os.path.exists(self.face_folder):
            os.makedirs(self.face_folder)
        # dlib 얼굴 검출기 및 랜드마크 예측기 초기화
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('C:/Project/Model_Integration/videos/shape_predictor_68_face_landmarks.dat')
        # 이전 랜드마크를 저장할 변수
        self.prev_landmarks = None
        # 얼굴 저장 카운터
        self.face_counter = 0
        # 프레임 간격 설정
        self.frame_interval = frame_interval

    def split_audio_and_video(self):
        # 비디오 클립 로드
        clip = VideoFileClip(self.input_video_path)
        # 오디오가 있는지 확인하고 오디오 파일로 저장
        if clip.audio:
            clip.audio.write_audiofile(self.output_audio_path, codec='pcm_s16le')
        else:
            print("No audio found in the video.")
        # 오디오가 없는 비디오 파일 저장
        clip.without_audio().write_videofile(self.output_video_path, codec='libx264')

    def process_video_frames(self):
        # 비디오 클립 로드
        clip = VideoFileClip(self.output_video_path)
        # 비디오의 총 길이 (프레임 수)
        total_frames = int(clip.fps * clip.duration)

        # 설정된 간격으로 프레임 처리
        for i in range(0, total_frames, self.frame_interval):
            frame = clip.get_frame(i / clip.fps)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) > 0:
                for face in faces:
                    # 랜드마크 검출
                    landmarks = self.predictor(gray, face)
                    # 얼굴 임시 저장
                    cropped_face_path = self.save_temp_face(frame_rgb, face, i)
                    # 랜드마크 시각화
                    self.visualize_landmarks(frame_rgb, landmarks)
                    # 표정 변화 감지
                    if self.detect_expression_change(landmarks):
                        # 표정 변화 로그 출력
                        print(f'Expression change detected at frame {i}')
                        # 얼굴 저장
                        self.save_face(frame_rgb, face)

    def detect_expression_change(self, current_landmarks):
        # 이전 랜드마크가 없으면 변화 없음
        if self.prev_landmarks is None:
            self.prev_landmarks = current_landmarks
            return False

        # 눈과 입 랜드마크 인덱스
        eye_indices = list(range(36, 48))  # 양쪽 눈
        mouth_indices = list(range(48, 68))  # 입

        prev_points = np.array([[self.prev_landmarks.part(i).x, self.prev_landmarks.part(i).y] for i in eye_indices + mouth_indices])
        curr_points = np.array([[current_landmarks.part(i).x, current_landmarks.part(i).y] for i in eye_indices + mouth_indices])

        # 눈과 입의 랜드마크 거리 계산하여 변화 감지
        expression_change = np.linalg.norm(prev_points - curr_points, axis=1).mean() > 15  # 임계값 높임

        if expression_change:
            self.prev_landmarks = current_landmarks  # 이전 랜드마크 업데이트

        return expression_change

    def save_temp_face(self, frame, box, frame_number):
        # 얼굴 영역 추출 (마진 포함)
        x1 = max(box.left() - self.margin, 0)
        y1 = max(box.top() - self.margin, 0)
        x2 = min(box.right() + self.margin, frame.shape[1])
        y2 = min(box.bottom() + self.margin, frame.shape[0])
        face_img = frame[y1:y2, x1:x2]
        temp_path = os.path.join(self.temp_folder, f'temp_face_{frame_number}.jpg')
        # 얼굴 이미지 임시 저장
        cv2.imwrite(temp_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        return temp_path

    def save_face(self, frame, box):
        # 얼굴 영역 추출 (마진 포함)
        x1 = max(box.left() - self.margin, 0)
        y1 = max(box.top() - self.margin, 0)
        x2 = min(box.right() + self.margin, frame.shape[1])
        y2 = min(box.bottom() + self.margin, frame.shape[0])
        face_img = frame[y1:y2, x1:x2]
        # 얼굴 이미지 저장 (숫자로 파일명 지정, 랜드마크 점 제거)
        save_path = os.path.join(self.face_folder, f'face_{self.face_counter}.jpg')
        cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        # 얼굴 카운터 증가
        self.face_counter += 1

    def visualize_landmarks(self, image, landmarks):
        # 원본 이미지를 복사하여 랜드마크를 시각화
        image_copy = image.copy()
        # 랜드마크 위치에 녹색 점 그리기
        for i in range(0, landmarks.num_parts):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(image_copy, (x, y), 2, (0, 255, 0), -1)
        # 이미지에 랜드마크 표시
        cv2.imshow("Landmarks", cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def clean_up(self):
        # 임시 폴더 삭제
        if os.path.exists(self.temp_folder):
            shutil.rmtree(self.temp_folder)

# 메인 실행
input_video_path = 'C:/Project/Model_Integration/videos/free_video4.mp4'
output_audio_path = 'C:/Project/Model_Integration/videos/output/wav/output_audio.wav'
output_video_path = 'C:/Project/Model_Integration/videos/output/mp4/output_video.mp4'
output_folder = 'C:/Project/Model_Integration/videos/output/faces'

video_processor = ProcessVideo(input_video_path, output_audio_path, output_video_path, output_folder)
video_processor.split_audio_and_video()
video_processor.process_video_frames()
video_processor.clean_up()

cv2.destroyAllWindows()
