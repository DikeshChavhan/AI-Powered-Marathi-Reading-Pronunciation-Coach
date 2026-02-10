import re
import whisper
import librosa
from jiwer import wer
from difflib import SequenceMatcher

# ----------------------------
# TEXT NORMALIZATION
# ----------------------------
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)  # Devanagari only
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# PRONUNCIATION SCORING
# ----------------------------
def pronunciation_score(expected_text, predicted_text):
    expected_norm = normalize_text(expected_text)
    predicted_norm = normalize_text(predicted_text)

    error_rate = wer(expected_norm, predicted_norm)
    raw_accuracy = (1 - error_rate) * 100
    accuracy = max(0, min(100, raw_accuracy))

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

    return accuracy, mistakes, expected_norm, predicted_norm


# ----------------------------
# FLUENCY SCORING
# ----------------------------
def fluency_score(audio_path, spoken_text):
    audio, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sr)

    words = spoken_text.split()
    word_count = len(words)

    wpm = (word_count / duration) * 60 if duration > 0 else 0
    return round(wpm, 2), round(duration, 2)


def fluency_level(wpm):
    if wpm < 60:
        return "Very Slow"
    elif wpm < 90:
        return "Slow"
    elif wpm < 120:
        return "Average"
    elif wpm < 150:
        return "Good"
    else:
        return "Excellent"


# ----------------------------
# MAIN PIPELINE
# ----------------------------
if __name__ == "__main__":

    # -------- INPUTS --------
    audio_file = "cv-corpus-24.0-2025-12-05/mr/clips/common_voice_mr_37991861.mp3"
    expected_text = "टिळकांनी गीतेचा कर्मयोग असा अर्थ लावला आहे."

    # -------- ASR --------
    print("\nLoading ASR model...")
    model = whisper.load_model("small")

    print("Transcribing audio...")
    result = model.transcribe(audio_file, language="mr")
    predicted_text = result["text"]

    # -------- PRONUNCIATION --------
    accuracy, mistakes, exp_norm, pred_norm = pronunciation_score(
        expected_text, predicted_text
    )

    # -------- FLUENCY --------
    wpm, duration = fluency_score(audio_file, pred_norm)
    fluency = fluency_level(wpm)

    # -------- OUTPUT --------
    print("\nEXPECTED TEXT:")
    print(exp_norm)

    print("\nPREDICTED TEXT:")
    print(pred_norm)

    print("\nPRONUNCIATION ACCURACY:")
    print(f"{accuracy:.2f}%")

    print("\nFLUENCY:")
    print(f"{wpm} WPM ({fluency})")
    print(f"Audio Duration: {duration} seconds")

    print("\nPRONUNCIATION MISTAKES:")
    if mistakes:
        for ref, hyp in mistakes:
            print(f"- Expected: '{ref}' → Spoken: '{hyp}'")
    else:
        print("No major pronunciation mistakes detected 🎉")
