import cv2
import os
import io
import json
import moviepy.editor as mp
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set your Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-service-account.json"

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

# Input video path
video_path = "cricket_match.mp4"
output_video = "cricket_highlights.mp4"
frames_dir = "frames"

# Create directory for frames
if not os.path.exists(frames_dir):
    os.makedirs(frames_dir)

# Extract frames from video
def extract_frames(video_path, frames_dir, interval=5):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count // fps
    print(f"Video FPS: {fps}, Total Duration: {duration} sec")

    frame_index = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_index % (fps * interval) == 0:
            frame_file = f"{frames_dir}/frame_{frame_index}.jpg"
            cv2.imwrite(frame_file, frame)
        frame_index += 1
    cap.release()
    print("Frames extracted successfully.")

# Analyze frame using Google Vision API
def analyze_frame(image_path):
    with io.open(image_path, "rb") as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    
    # Run OCR (text detection)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    detected_text = texts[0].description if texts else ""

    # Run object detection
    response = client.object_localization(image=image)
    objects = [obj.name for obj in response.localized_object_annotations]
    
    return detected_text, objects

# Process all frames and identify highlights
def identify_highlights(frames_dir):
    highlight_frames = []
    prev_score = ""

    for frame_file in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_file)
        text, objects = analyze_frame(frame_path)

        # Detect scoreboard change
        if "Runs" in text or "Score" in text:
            if text != prev_score:  # Score change detected
                print(f"Score Change Detected: {text}")
                highlight_frames.append(frame_path)
                prev_score = text
        
        # Detect important events (e.g., players, bat, ball)
        if any(event in objects for event in ["Person", "Bat", "Ball"]):
            highlight_frames.append(frame_path)

    return list(set(highlight_frames))

# Generate highlight video
def create_highlight_video(video_path, highlight_frames, output_video):
    video = mp.VideoFileClip(video_path)
    clips = []

    for frame in highlight_frames:
        frame_number = int(frame.split("_")[-1].split(".")[0])
        start_time = frame_number / video.fps
        clip = video.subclip(start_time, start_time + 5)  # Take 5-second clips
        clips.append(clip)

    final_clip = mp.concatenate_videoclips(clips)
    final_clip.write_videofile(output_video, codec="libx264")
    print(f"Highlights saved to {output_video}")

# Run the pipeline
extract_frames(video_path, frames_dir)
highlight_frames = identify_highlights(frames_dir)
create_highlight_video(video_path, highlight_frames, output_video)
