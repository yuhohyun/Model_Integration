import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일에서 환경 변수 로드
load_dotenv()

OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

# 음성 파일 경로
audio_file = open("C:/Project/Model_Integration/audio/sample/000020.wav", "rb")

transcript = client.audio.transcriptions.create(
  file=audio_file,
  model="whisper-1",
  language="ko",
  response_format="json",
  timestamp_granularities=["segment"]
)

print(transcript.text)