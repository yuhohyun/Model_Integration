from moviepy.editor import VideoFileClip

def split_audio_and_video(input_video_path, output_audio_path, output_video_path):
    # 비디오 파일 로드
    clip = VideoFileClip(input_video_path)

    # 오디오 추출 및 저장
    if clip.audio:
        clip.audio.write_audiofile(output_audio_path, codec='pcm_s16le')  # WAV 형식으로 저장
    else:
        print("No audio found in the video.")

    # 비디오에서 오디오 제거 및 저장
    clip.without_audio().write_videofile(output_video_path, codec='libx264')

# 비디오 경로 입력
input_video_path = 'free_video.mp4'
output_audio_path = 'output_audio.wav'
output_video_path = 'output_video.mp4'

# split_audio_and_video 함수 호출
split_audio_and_video(input_video_path, output_audio_path, output_video_path)