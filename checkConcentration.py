import cv2
import numpy as np
from gaze_tracking import GazeTracking

# Initialize gaze tracking
def initialize_gaze_tracking():
    gaze = GazeTracking()
    return gaze

# Process frame for gaze tracking
def process_gaze_tracking(frame, gaze):
    gaze.refresh(frame)
    is_gaze_focused = gaze.is_right() or gaze.is_left() or gaze.is_center()
    return is_gaze_focused

# Calculate the concentration ratio
def calculate_concentration_ratio(total_frames, focused_frames):
    concentration_ratio = focused_frames / total_frames if total_frames > 0 else 0
    return concentration_ratio

# Print the progress of video processing
def print_progress(current_frame, total_frames):
    progress_percentage = (current_frame / total_frames) * 100
    print(f"Processing video: {progress_percentage:.2f}% complete", end='\r')

def main(video_path):
    gaze = initialize_gaze_tracking()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    focused_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Cannot open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:  # Process only every 5th frame
            is_gaze_focused = process_gaze_tracking(frame, gaze)
            if is_gaze_focused:
                focused_count += 1
            print_progress(frame_count, total_frames)

        frame_count += 1

    concentration_ratio = calculate_concentration_ratio(frame_count // 5, focused_count)
    print(f"\nConcentration Ratio: {concentration_ratio:.2f} (Focused: {focused_count}, Total Processed: {frame_count // 5})")

    cap.release()

if __name__ == "__main__":
    video_path = "C:\\Project\\Model_Integration\\videos\\gaze.MOV"
    main(video_path)
