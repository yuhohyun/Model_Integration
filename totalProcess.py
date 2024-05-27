import os
import cv2
import numpy as np
import dlib
import nltk
import torch
from moviepy.editor import VideoFileClip
from PIL import Image
import torchvision.transforms as transforms
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
from models.image_classification.model import modified_PAtt_Lite
from transformers import ElectraModel, ElectraTokenizer, PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration
from models.text_classification.model import CustomTransformerModel

# .env 파일에서 환경 변수 로드
load_dotenv()

# NLTK 문장 토크나이저 리소스 다운로드
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# dlib face detector와 landmark detector 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Project/Model_Integration/videos/shape_predictor_68_face_landmarks.dat')

# 텍스트 요약을 위한 미세 조정된 BART 모델 로드
model_finetuned = BartForConditionalGeneration.from_pretrained('./models/text_summarization')

# 텍스트 요약 모델의 토크나이저 로드
tokenizer_summarization = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

# 텍스트 분류 모델의 토크나이저 로드
tokenizer_text_classification = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 텍스트 분류를 위한 사전 학습된 Electra 모델 로드
pretrained_text_classification_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

# 분류를 위한 레이블 수와 분류 벡터의 크기 정의
num_labels = 3
num_cls_vector = pretrained_text_classification_model.config.hidden_size

# 사전 학습된 Electra 모델을 사용하여 커스텀 텍스트 분류 모델 초기화
text_classification_model = CustomTransformerModel(pretrained_text_classification_model, num_labels, num_cls_vector)

# 파일에서 텍스트 분류 모델의 상태 사전 로드
state_dict = torch.load('models/text_classification/model_textClassification.pth', map_location=torch.device('cpu'))

# 상태 사전에서 position_ids 키가 존재할 경우 제거 (이 키는 필요하지 않을 수 있음)
state_dict.pop("model.embeddings.position_ids", None)

# 텍스트 분류 모델에 상태 사전 로드
text_classification_model.load_state_dict(state_dict)
# 모델을 평가 모드로 설정
text_classification_model.eval()

# 이미지 분류 모델 로드
image_model = modified_PAtt_Lite(num_classes=3, pretrained=False)
image_model.load_state_dict(torch.load('models/image_classification/model_imageClassification.pth', map_location=torch.device('cpu')))
# 평가 모드로 설정
image_model.eval()


def split_audio(audio_file):
    # 주어진 음성 파일을 로드
    audio = AudioSegment.from_file(audio_file)
    
    # 1분(60초 * 1000밀리초)을 밀리초 단위로 정의
    minute = 60 * 1000
    
    # 음성 파일을 1분 단위로 분할하여 리스트에 저장
    # 음성 파일의 길이만큼 반복하여 1분씩 분할
    parts = [audio[i:i + minute] for i in range(0, len(audio), minute)]
    
    # 분할된 음성 파일 조각들을 반환
    return parts

def whisper(audio_file_path, output_dir="temp", model="whisper-1", language="ko"):
    
    # API 키 로드
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI()
    
    # 음성 파일 분할
    audio_parts = split_audio(audio_file_path)
    
    # 결과 저장할 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 텍스트 변환 결과를 저장할 리스트 초기화
    transcripts = []
    
    # 분할된 각 음성 파일 부분을 텍스트로 변환
    for index, part in enumerate(audio_parts):
        # 현재 음성 부분을 임시 파일로 저장
        part_file_path = os.path.join(output_dir, f"part_{index}.wav")
        part.export(part_file_path, format="wav")

        # API 호출을 위해 임시 파일을 열기
        with open(part_file_path, "rb") as file:
            transcript = client.audio.transcriptions.create(
                file=file,
                model=model,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
            
            # 변환된 텍스트를 리스트에 저장
            transcripts.append(transcript.text)

        # 사용 후 임시 파일 삭제
        os.remove(part_file_path)
    
    # 모든 변환된 텍스트 반환
    return transcripts

def summarizeTexts(transcripts):
    # 요약된 텍스트를 저장할 리스트 초기화
    summaries = []
    
    # 각 변환된 텍스트에 대해 요약 수행
    for text in transcripts:
        if text:
            # 텍스트를 토크나이저로 인코딩
            input_ids = tokenizer_summarization.encode(text)
            # 인코딩된 텍스트를 텐서로 변환
            input_ids = torch.tensor(input_ids)
            # 배치 차원을 추가하여 모델 입력 형식에 맞춤
            input_ids = input_ids.unsqueeze(0)
            # 미세 조정된 모델을 사용하여 요약 생성
            output = model_finetuned.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
            # 생성된 요약을 디코딩하여 텍스트로 변환
            summary = tokenizer_summarization.decode(output[0], skip_special_tokens=True)
            # 요약된 텍스트를 리스트에 추가
            summaries.append(summary)
    
    # 모든 요약된 텍스트 반환
    return summaries

def infer_text_classification(text):
    # 텍스트를 입력으로 받아 토크나이저로 인코딩
    inputs = tokenizer_text_classification(text, return_tensors='pt', padding=True, truncation=True)
    
    # token_type_ids가 입력에 포함되어 있으면 제거
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    
    # 입력 데이터를 딕셔너리 형태로 변환
    inputs = {k: v for k, v in inputs.items()}
    
    # 모델을 사용하여 예측 수행 (평가 모드에서 그래디언트 계산 비활성화)
    with torch.no_grad():
        outputs = text_classification_model(**inputs)
    
    # 모델의 출력인 로짓(logits) 값을 얻음
    logits = outputs
    # 소프트맥스 함수를 적용하여 확률 계산
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # 확률이 가장 높은 클래스의 인덱스를 얻음
    class_idx = torch.argmax(probs, dim=-1).item()
    
    # 클래스 레이블 정의
    labels = ['negative', 'neutral', 'positive']
    # 예측된 클래스 레이블 반환
    return labels[class_idx]

def classify_sentences(transcripts):
    # 분류 결과를 저장할 리스트 초기화
    classification_results = []
    
    # 각 변환된 텍스트에 대해 문장 단위로 분류 수행
    for transcript in transcripts:
        # 텍스트를 문장 단위로 분할
        sentences = sent_tokenize(transcript)
        for sentence in sentences:
            # 각 문장에 대해 텍스트 분류 수행
            classification = infer_text_classification(sentence)
            # 분류 결과를 리스트에 추가
            classification_results.append(classification)
    
    # 모든 분류 결과 반환
    return classification_results

def detect_expression_change(current_landmarks, prev_landmarks):
    # 이전 랜드마크가 없다면, 변화를 감지할 수 없으므로 False 반환
    if prev_landmarks is None:
        return False, current_landmarks

    # 눈과 입에 해당하는 랜드마크 인덱스
    eye_indices = list(range(36, 48))  # 양쪽 눈
    mouth_indices = list(range(48, 68))  # 입

    # 이전과 현재의 랜드마크 포인트를 numpy 배열로 추출
    prev_points = np.array([[prev_landmarks.part(i).x, prev_landmarks.part(i).y] for i in eye_indices + mouth_indices])
    curr_points = np.array([[current_landmarks.part(i).x, current_landmarks.part(i).y] for i in eye_indices + mouth_indices])

    # 두 랜드마크 사이의 유클리디안 거리를 계산하고, 평균이 특정 임계값보다 크면 표정 변화로 판단
    expression_change = np.linalg.norm(prev_points - curr_points, axis=1).mean() > 15

    # 표정 변화가 있었는지 여부와 현재 랜드마크를 반환
    return expression_change, current_landmarks

def process_video_frames(output_video_path, frame_interval=5):
    # 비디오 클립을 로드하고 총 프레임 수 계산
    clip = VideoFileClip(output_video_path)
    total_frames = int(clip.fps * clip.duration)
    
    # 표정 변화가 감지된 프레임을 저장할 리스트 초기화
    changed_frames = []
    
    # 이전 프레임의 랜드마크 초기화
    prev_landmarks = None

    # 설정된 간격으로 프레임을 검사
    for i in range(0, total_frames, frame_interval):
        # 해당 시점의 프레임을 추출
        frame = clip.get_frame(i / clip.fps)
        # 프레임을 RGB 색공간으로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # RGB 프레임을 그레이스케일로 변환
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
        # 그레이스케일 이미지에서 얼굴 검출
        faces = detector(gray)

        # 감지된 각 얼굴에 대하여
        for face in faces:
            # 얼굴에서 랜드마크 추출
            landmarks = predictor(gray, face)
            # 이전 랜드마크와 비교하여 표정 변화 감지
            expression_change, prev_landmarks = detect_expression_change(landmarks, prev_landmarks)
            # 표정 변화가 있었다면
            if expression_change:
                print(f'Expression change detected at frame {i}')
                # 표정 변화가 감지된 프레임을 리스트에 추가
                changed_frames.append(frame_rgb)

    return changed_frames

def crop_faces_with_margin(changed_frames, detector, margin=20):
    # 크롭된 얼굴 이미지를 저장할 리스트 초기화
    cropped_faces = []
    
    # 감지된 각 프레임에 대해 반복
    for frame in changed_frames:
        # 프레임을 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 그레이스케일 이미지에서 얼굴 검출
        faces = detector(gray)

        # 감지된 각 얼굴에 대하여
        for face in faces:
            # 얼굴 영역 주변에 마진을 포함하여 좌표 계산
            x1 = max(face.left() - margin, 0)
            y1 = max(face.top() - margin, 0)
            x2 = min(face.right() + margin, frame.shape[1])
            y2 = min(face.bottom() + margin, frame.shape[0])
            # 계산된 좌표로 얼굴 영역을 크롭
            cropped_face = frame[y1:y2, x1:x2]
            # 크롭된 얼굴 이미지를 리스트에 추가
            cropped_faces.append(cropped_face)

    # 크롭된 얼굴 이미지의 리스트 반환
    return cropped_faces

def preprocess_face(face):
    """단일 얼굴 이미지를 전처리하는 함수"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Numpy 배열 (OpenCV 이미지)을 PIL 이미지로 변환
    face_image = Image.fromarray(face)
    face_image = transform(face_image)
    return face_image.unsqueeze(0)  # 배치 차원 추가

def infer_image_classification(model, cropped_faces):
    """크롭된 얼굴 이미지를 순차적으로 표정 분류 추론을 수행하는 함수"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for face in cropped_faces:
            image = preprocess_face(face)
            output = model(image)
            _, prediction = torch.max(output, 1)
            predictions.append(prediction.item())  # 각 이미지에 대한 분류 결과 저장
    return predictions

def audioProcess(audioPath, outputDir):
    # 음성 파일을 텍스트로 변환
    transcripts = whisper(audioPath, outputDir)
    # 변환된 텍스트를 요약
    summaries = summarizeTexts(transcripts)
    # 요약된 텍스트를 감정 분석 모델에 입력하여 분류
    classification_results = classify_sentences(transcripts)
    # 요약문 및 감정 분석 결과 출력
    for i, summary in enumerate(summaries):
        print(f"Summary {i + 1}:")
        print(summary)
        print("===================================")
    # 감정 분석 결과 출력
    print("Classification Results:")
    for i, result in enumerate(classification_results):
        print(f"Sentence {i + 1}: {result}")
        
def videoProcess(videoPath):
    # 비디오에서 표정 변화가 감지된 프레임 추출
    changed_frames = process_video_frames(videoPath)
    # 표정 변화가 감지된 프레임에서 얼굴을 크롭
    cropped_faces = crop_faces_with_margin(changed_frames, detector, 20)
    # 크롭된 얼굴 이미지를 바탕으로 이미지 분류
    predictions = infer_image_classification(image_model, cropped_faces)
    for i, prediction in enumerate(predictions):
        if prediction == 0:
            prediction_str = 'negative'
        elif prediction == 1:
            prediction_str = 'neutral'
        elif prediction == 2:
            prediction_str = 'positive'
        else:
            prediction_str = 'Unknown'
        print(f'Image {i + 1} prediction: {prediction_str}')
    return predictions
        
def getScore(classification_results) :
    
    score = 0

    for i in classification_results :
        if i == 'neutral' :
            score += 1
        if i == 'negative' :
            continue
        if i == 'positive' :
            score += 2
    return score / len(classification_results) * 50

def getImageScore(predictions) :
    return sum(predictions) / len(predictions) * 50

def main():
    # 음성 파일 경로 및 출력 디렉토리 설정
    audio = "C:/Project/Model_Integration/audio/sample/combined.wav"
    output = "C:/Project/Model_Integration/audio/temp"
    # 비디오 파일 경로 설정
    video = "C:/Project/Model_Integration/videos/free_video4.mp4"
    
    # 음성 파일을 텍스트로 변환 후, 요약 및 감정 분석을 수행하는 함수
    audioProcess(audio, output)
    # 비디오 파일 속 얼굴을 검출해 감정 분석을 수행하는 함수
    videoProcess(video)


# main 함수 호출
if __name__ == "__main__":
    main()
 