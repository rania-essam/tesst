import os
import cv2
import numpy as np
import tensorflow as tf
import librosa
import pandas as pd
from joblib import load
from moviepy.editor import VideoFileClip
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.responses import JSONResponse

app = FastAPI()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

saved_model = tf.keras.models.load_model("./lstm_without_masking(94).h5")
audio_model = tf.keras.models.load_model("./audio_model.h5")
scaler = load("./scaler.joblib")

latest_result = None

# Function to validate input video (one person, clear face)
def input_validation(input_video):
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()
    cap = cv2.VideoCapture(input_video)

    valid = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if not results.detections or len(results.detections) > 1:
            valid = False
            break
    cap.release()
    return valid

# Function to check if the video has audio
def check_audio_in_video(video_file):
    video = VideoFileClip(video_file)
    return 1 if video.audio else 0

# Extract audio from video
def extract_audio_from_video(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file, codec='pcm_s16le')

# Split audio into segments
def split_and_pad_audio(audio_file, segment_duration=10, sample_rate=16000):
    segment_samples = segment_duration * sample_rate
    audio, _ = librosa.load(audio_file, sr=sample_rate)
    num_segments = len(audio) // segment_samples
    remainder = len(audio) % segment_samples
    segments = [audio[i * segment_samples : (i + 1) * segment_samples] for i in range(num_segments)]
    if remainder > 0:
        last_segment = np.pad(audio[num_segments * segment_samples:], (0, segment_samples - remainder), mode='constant')
        segments.append(last_segment)
    return segments

# Extract MFCC features
def extract_mfcc_from_segments(segments, sample_rate=16000, n_mfcc=13):
    features = []
    for segment in segments:
        mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        features.append(np.concatenate([mfcc_mean, mfcc_std]))
    return np.array(features)

# Predict audio lie/truth
def make_prediction_audio(input_video):
    if not check_audio_in_video(input_video):
        return [1]  # Default to truth if no audio
    audio_file = "extracted_audio.wav"
    extract_audio_from_video(input_video, audio_file)
    segments = split_and_pad_audio(audio_file)
    mfcc_features = extract_mfcc_from_segments(segments)
    mfcc_features = scaler.transform(mfcc_features)
    mfcc_features = np.expand_dims(mfcc_features, axis=1)
    predictions = audio_model.predict(mfcc_features)
    return 1 - np.argmax(predictions, axis=1)

# Predict facial lie/truth
def make_prediction(input_video):
    video_capture = cv2.VideoCapture(input_video)
    if not video_capture.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video")
    if not input_validation(input_video):
        raise HTTPException(status_code=400, detail="Video should contain one person with a clear face")
    predictions = []
    frame_count = 0
    cur_video32 = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_count += 1
        cur_video32.append(np.zeros(478 * 3))  # Placeholder since face mesh isn't implemented here
        if frame_count == 32:
            predictions.append(np.argmax(saved_model.predict(np.array([cur_video32]))))
            cur_video32 = []
            frame_count = 0
    video_capture.release()
    return predictions

# Final decision function
def final_decision(input_video):
    global latest_result
    voice_preds = make_prediction_audio(input_video)
    face_preds = make_prediction(input_video)
    voice_preds = np.repeat(voice_preds, 10)[:len(face_preds)]
    face_preds = np.pad(face_preds, (0, max(0, len(voice_preds) - len(face_preds))), constant_values=1)
    hard_vote = (voice_preds + face_preds) >= 1
    final_result = "Lie" if np.sum(hard_vote) > len(hard_vote) / 2 else "Truth"
    confidence = (max(np.sum(hard_vote), len(hard_vote) - np.sum(hard_vote)) / len(hard_vote)) * 100
    latest_result = {"final_decision": final_result, "confidence": confidence}
    return latest_result

@app.post("/process_video")
async def process_video(video: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, video.filename)
    with open(file_path, "wb") as f:
        f.write(await video.read())
    result = final_decision(file_path)
    return JSONResponse(content=result)

@app.get("/result")
async def get_result():
    if latest_result is None:
        raise HTTPException(status_code=404, detail="No analysis performed yet")
    return JSONResponse(content=latest_result)
