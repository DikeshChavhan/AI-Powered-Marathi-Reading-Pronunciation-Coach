import streamlit as st
import whisper
import librosa
import os
import numpy as np
from jiwer import wer
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import queue

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Marathi Reading & Pronunciation Coach",
    page_icon="📖",
    layout="centered"
)

st.title("📖 AI-Powered Marathi Reading & Pronunciation Coach")
st.write("Record or upload Marathi reading audio and get pronunciation & fluency feedback.")

# -----------------------------
# Expected Text
# -----------------------------
st.subheader("📘 Expected Text")
expected_text = st.text_area(
    "",
    "टिळकांनी गीतेचा कर्मयोग असा अर्थ लावला आहे."
)

# -----------------------------
# Audio Recorder (Mic)
# -----------------------------
st.subheader("🎤 Read Aloud (Mic Recording)")

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = queue.Queue()

    def recv(self, frame: av.AudioFrame):
        self.audio_frames.put(frame)
        return frame

ctx = webrtc_streamer(
    key="marathi-reader",
    audio_processor_factory=AudioRecorder,
    media_stream_constraints={"audio": True, "video": False},
)

audio_path = None

if ctx.audio_processor:
    if st.button("⏹ Stop & Save Recording"):
        frames = []
        while not ctx.audio_processor.audio_frames.empty():
            frame = ctx.audio_processor.audio_frames.get()
            frames.append(frame.to_ndarray())

        if frames:
            audio = np.concatenate(frames, axis=1)
            audio_path = "recorded_audio.wav"
            librosa.output.write_wav(audio_path, audio.flatten(), sr=16000)
            st.success("Recording saved successfully!")
            st.audio(audio_path)

# -----------------------------
# Upload Audio Option
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Or Upload Audio (MP3 / WAV)",
    type=["mp3", "wav"]
)

if uploaded_file is not None:
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(audio_path)

# -----------------------------
# Analyze
# -----------------------------
if st.button("🧠 Analyze Reading"):
    if audio_path is None:
        st.warning("Please record or upload audio first.")
        st.stop()

    with st.spinner("Loading ASR model..."):
        model = whisper.load_model("small")

    with st.spinner("Transcribing audio..."):
        result = model.transcribe(audio_path, language="mr")
        predicted_text = result["text"].strip()

    st.subheader("📝 Predicted Text")
    st.write(predicted_text)

    # -----------------------------
    # Pronunciation Accuracy
    # -----------------------------
    def normalize(text):
        return text.replace("।", "").strip()

    expected_norm = normalize(expected_text)
    predicted_norm = normalize(predicted_text)

    try:
        error_rate = wer(expected_norm, predicted_norm)
        accuracy = max(0, round((1 - error_rate) * 100, 2))
    except:
        accuracy = 0.0

    st.subheader("🎯 Pronunciation Accuracy")
    st.metric("Accuracy", f"{accuracy}%")

    # -----------------------------
    # Fluency
    # -----------------------------
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    wpm = round((len(predicted_norm.split()) / duration) * 60, 2)

    st.subheader("⏱ Fluency")
    st.write(f"**Words Per Minute:** {wpm}")
    st.write(f"**Audio Duration:** {round(duration, 2)} seconds")

    # Cleanup
    os.remove(audio_path)
