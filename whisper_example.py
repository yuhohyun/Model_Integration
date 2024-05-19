import os
import nltk
import time
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

# NLTK 문장 토크나이저 리소스 다운로드
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# .env 파일에서 환경 변수 로드
load_dotenv()

# API KEY(.env 에 저장되어 있음)
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# 음성 파일 로드
audio_file_path = "C:/Project/Model_Integration/audio/sample/combined.wav"
audio = AudioSegment.from_file(audio_file_path)

# 음성 파일을 1분(60초 * 1000밀리초) 단위로 분할
minute = 60 * 1000
parts = [audio[i:i + minute] for i in range(0, len(audio), minute)]

# 결과 저장할 디렉토리 생성
os.makedirs("C:/Project/Model_Integration/transcripts/for_classification", exist_ok=True)
os.makedirs("C:/Project/Model_Integration/transcripts/for_summary", exist_ok=True)

# 시간 측정 시작
start_time = time.time()

# 분할된 각 부분을 텍스트로 변환
for index, part in enumerate(parts):
    # 임시 파일로 저장
    part_file_path = f"C:/Project/Model_Integration/audio/temp/part_{index}.wav"
    part.export(part_file_path, format="wav")

    # API 호출을 위한 파일 열기
    with open(part_file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
        file=f,
        model="whisper-1",
        language="ko",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
        )
        
        print(transcript.text)
        
        # 전체 텍스트를 저장 (Summary 용)
        summary_txt_file_path = f"C:/Project/Model_Integration/transcripts/for_summary/part_{index}.txt"
        with open(summary_txt_file_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(transcript.text)

        # 문장별로 텍스트를 저장 (Classification 용)
        sentences = sent_tokenize(transcript.text)
        classification_txt_file_path = f"C:/Project/Model_Integration/transcripts/for_classification/part_{index}.txt"
        with open(classification_txt_file_path, "w", encoding="utf-8") as txt_file:
            for sentence in sentences:
                txt_file.write(sentence + "\n")

    # 사용 후 임시 파일 삭제
    os.remove(part_file_path)
   
    print(f"Part {index + 1} transcribed and saved to {summary_txt_file_path} and {classification_txt_file_path}")

# 실행 시간 계산 및 출력
end_time = time.time()
elapsed_time = end_time - start_time
print("===================================================")
print(f"Total processing time: {elapsed_time} seconds")