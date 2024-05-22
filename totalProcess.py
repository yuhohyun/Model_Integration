import os
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
import nltk
import torch
from transformers import ElectraModel, ElectraTokenizer, PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration
from models.text_classification.model import CustomTransformerModel

# NLTK 문장 토크나이저 리소스 다운로드
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# .env 파일에서 환경 변수 로드
load_dotenv()

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

def main():
    # 음성 파일 경로 및 출력 디렉토리 설정
    audio_file_path = "C:/Project/Model_Integration/audio/sample/combined.wav"
    output_directory = "C:/Project/Model_Integration/audio/temp"
    
    # 음성 파일을 텍스트로 변환
    transcripts = whisper(audio_file_path, output_directory)
    
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

# main 함수 호출
if __name__ == "__main__":
    main()
