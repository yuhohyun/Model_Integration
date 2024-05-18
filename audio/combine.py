from pydub import AudioSegment
import os

# 폴더 경로 설정
folder_path = 'C:/project/Model_Integration/audios/sample'  # 폴더 경로 지정

# 모든 wav 파일 리스트 생성
wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# 첫 번째 파일을 기본 오디오 트랙으로 로드
combined = AudioSegment.from_file(os.path.join(folder_path, wav_files[0]))

# 나머지 파일을 순차적으로 결합
for wav_file in wav_files[1:]:
    next_audio = AudioSegment.from_file(os.path.join(folder_path, wav_file))
    combined += next_audio

# 결합된 오디오를 파일로 저장
combined.export(os.path.join(folder_path, "combined.wav"), format='wav')
