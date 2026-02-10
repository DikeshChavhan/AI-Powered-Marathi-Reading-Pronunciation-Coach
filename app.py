import streamlit as st
import whisper
import soundfile as sf
import librosa
import os
import numpy as np
from jiwer import wer
from st_audiorec import st_audiorec

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI-Powered Marathi Reading & Pronunciation Coach",
    page_icon="📖",
    layout="centered"
)

st.title("📖 AI-Powered Marathi Reading & Pronunciation Coach")
st.write("Upload or record Marathi reading audio and get pronunciation & fluency feedback.")

# -------------------------------
# Expected Text
# -------------------------------
st.subheader("📘 Expected Text (What the student should read)")
expected_text = st.text_area(
    "",
    "टिळकांनी गीतेचा कर्मयोग असा अर्थ लावला आहे."
)

# -------------------------------
# Audio Input Options
# -------------------------------
st.subheader("🎤 Read Aloud (Mic Recording)")
mic_audio = st_audiorec()

uploaded_file = st.file_uploader(
    "📂 Or Upload Audio (MP3 / WAV)",
    type=["mp3", "wav"]
)

audio_path = None

# -------------------------------
# Handle Mic Audio
# -------------------------------
if mic_audio is not None:
    audio_path = "recorded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(mic_audio)
    st.audio(audio_path)

# -------------------------------
# Handle Uploaded Audio
# -------------------------------
elif uploaded_file is not None:
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(audio_path)

# -------------------------------
# Analyze Button
# -------------------------------
if st.button("🧠 Analyze Reading"):

    if audio_path is None:
        st.warning("Please record or upload audio first.")
        st.stop()

    # Load ASR model
    with st.spinner("Loading ASR model..."):
        model = whisper.load_model("small")

    # Transcribe
    with st.spinner("Transcribing audio..."):
        result = model.transcribe(audio_path, language="mr")
        predicted_text = result["text"].strip()

    st.subheader("📝 Predicted Text")
    st.write(predicted_text)

    # -------------------------------
    # Pronunciation Accuracy
    # -------------------------------
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
    st.metric(label="Accuracy", value=f"{accuracy}%")

    # -------------------------------
    # Fluency (WPM)
    # -------------------------------
    y, sr = librosa.load(audio_path, sr=None)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    word_count = len(predicted_norm.split())

    wpm = round((word_count / duration_sec) * 60, 2) if duration_sec > 0 else 0

    st.subheader("⏱ Fluency")
    st.write(f"**Words Per Minute (WPM):** {wpm}")
    st.write(f"**Audio Duration:** {round(duration_sec, 2)} seconds")

    # -------------------------------
    # Interpretation
    # -------------------------------
    if wpm < 60:
        fluency_level = "Very Slow"
    elif wpm < 90:
        fluency_level = "Slow"
    elif wpm < 130:
        fluency_level = "Good"
    else:
        fluency_level = "Excellent"

    st.info(f"📊 Fluency Level: **{fluency_level}**")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)
