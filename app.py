import streamlit as st
import whisper
import librosa
import os
from jiwer import wer

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Marathi Reading & Pronunciation Coach",
    page_icon="📖",
    layout="centered"
)

st.title("📖 AI-Powered Marathi Reading & Pronunciation Coach")
st.write("Upload a Marathi reading audio and get pronunciation & fluency feedback.")

# -----------------------------
# Expected Text
# -----------------------------
st.subheader("📘 Expected Text (What the student should read)")
expected_text = st.text_area(
    "",
    "टिळकांनी गीतेचा कर्मयोग असा अर्थ लावला आहे."
)

# -----------------------------
# Upload Audio
# -----------------------------
st.subheader("🎤 Upload Audio (MP3 / WAV)")
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav"]
)

audio_path = None

if uploaded_file:
    audio_path = "student_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())
    st.audio(audio_path)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button("🧠 Analyze Reading"):

    if audio_path is None:
        st.warning("Please upload an audio file first.")
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
    st.write(f"Words Per Minute (WPM): {wpm}")
    st.write(f"Audio Duration: {round(duration, 2)} seconds")

    # Cleanup
    os.remove(audio_path)
