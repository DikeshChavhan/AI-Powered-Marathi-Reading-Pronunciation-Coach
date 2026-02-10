import streamlit as st
import whisper
import librosa
import numpy as np
import re
from jiwer import wer
from difflib import SequenceMatcher

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)  # Devanagari only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pronunciation_score(expected_text, predicted_text):
    expected_norm = normalize_text(expected_text)
    predicted_norm = normalize_text(predicted_text)

    error_rate = wer(expected_norm, predicted_norm)
    raw_accuracy = (1 - error_rate) * 100
    accuracy = max(0, min(100, raw_accuracy))

    # Accuracy → Level
    if accuracy < 30:
        level = "Needs Improvement"
    elif accuracy < 60:
        level = "Developing"
    elif accuracy < 80:
        level = "Good"
    else:
        level = "Excellent"

    # Word-level differences
    ref_words = expected_norm.split()
    hyp_words = predicted_norm.split()

    matcher = SequenceMatcher(None, ref_words, hyp_words)
    mistakes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            mistakes.append((
                " ".join(ref_words[i1:i2]),
                " ".join(hyp_words[j1:j2])
            ))

    return accuracy, level, mistakes, expected_norm, predicted_norm


def fluency_score(audio_array, sr, spoken_text):
    duration = librosa.get_duration(y=audio_array, sr=sr)
    words = spoken_text.split()
    wpm = (len(words) / duration) * 60 if duration > 0 else 0

    if wpm < 60:
        fluency = "Very Slow"
    elif wpm < 90:
        fluency = "Slow"
    elif wpm < 120:
        fluency = "Average"
    elif wpm < 150:
        fluency = "Good"
    else:
        fluency = "Excellent"

    return round(wpm, 2), fluency, round(duration, 2)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Powered Marathi Reading Coach",
    layout="centered"
)

st.title("📖 AI-Powered Marathi Reading & Pronunciation Coach")
st.write(
    "Upload a Marathi reading audio and get **pronunciation** and **fluency** feedback."
)

expected_text = st.text_area(
    "📘 Expected Text (What the student should read)",
    "टिळकांनी गीतेचा कर्मयोग असा अर्थ लावला आहे."
)

audio_file = st.file_uploader(
    "🎤 Upload Audio (MP3 / WAV)",
    type=["mp3", "wav"]
)

if audio_file:
    # Load audio safely (NO ffmpeg required)
    audio_array, sr = librosa.load(audio_file, sr=16000)

    st.audio(audio_file)

    if st.button("🧠 Analyze Reading"):
        with st.spinner("Loading ASR model..."):
            model = whisper.load_model("small")

        with st.spinner("Transcribing audio..."):
            result = model.transcribe(
                audio_array,
                language="mr",
                temperature=0.0,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                no_speech_threshold=0.2,
                logprob_threshold=-1.0
            )

        predicted_text = result["text"].strip()

        accuracy, level, mistakes, exp_norm, pred_norm = pronunciation_score(
            expected_text, predicted_text
        )

        wpm, fluency, duration = fluency_score(audio_array, sr, pred_norm)

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        st.subheader("📝 Transcription")
        st.write(pred_norm if pred_norm else "_No clear speech detected_")

        st.subheader("🎯 Pronunciation Result")
        st.write(f"**Level:** {level}")
        st.write(f"**Accuracy:** {accuracy:.2f}%")

        st.subheader("⏱ Fluency")
        st.write(f"**{wpm} WPM** ({fluency})")
        st.write(f"Audio Duration: {duration} seconds")

        st.subheader("❌ Pronunciation Mistakes")
        if len(pred_norm.split()) < 3:
            st.warning("Please read the full sentence clearly.")
        elif mistakes:
            for ref, hyp in mistakes:
                st.write(f"- Expected: **{ref}** → Spoken: **{hyp}**")
        else:
            st.success("Great pronunciation! 🎉")
