from dotenv import load_dotenv
import os
import requests
from pydub import AudioSegment
import json

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
lang = "Kor"  # 언어 코드 (Kor, Jpn, Eng, Chn)
url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang

# 오디오 파일 로드
audio = AudioSegment.from_file('C:/Project/Model_Integration/audio/sample/combined.wav')
duration = len(audio)  # 오디오 길이 (밀리초 단위)
one_minute = 60 * 1000  # 1분은 60000 밀리초

# 1분 단위로 오디오 분할
parts = [audio[i:i + one_minute] for i in range(0, duration, one_minute)]

# 파일로 저장할 준비
output_file = 'output_text.txt'
with open(output_file, 'w', encoding='utf-8') as file:  # 파일을 쓰기 모드로 열기

    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/octet-stream"
    }

    for index, part in enumerate(parts):
        # 파트를 임시 파일로 저장
        part_path = f'temp_part_{index}.wav'
        part.export(part_path, format="wav")
        
        # API 요청
        with open(part_path, 'rb') as data:
            response = requests.post(url, data=data, headers=headers)
            rescode = response.status_code

            if rescode == 200:
                response_json = json.loads(response.text)
                text = response_json['text']
                print(f"Part {index + 1}: {text}")
                file.write(f"Part {index + 1}: {text}\n")  # 파일에 텍스트 저장
            else:
                print(f"Part {index + 1} Error: {response.text}")
                file.write(f"Part {index + 1} Error: {response.text}\n")

        # 임시 파일 삭제
        os.remove(part_path)
