import librosa

# ----------------------------
# Calculate Fluency (WPM)
# ----------------------------
def calculate_fluency(audio_path, spoken_text):
    # Load audio to get duration
    audio, sr = librosa.load(audio_path, sr=None)
    duration_seconds = librosa.get_duration(y=audio, sr=sr)

    # Count words spoken
    words = spoken_text.strip().split()
    word_count = len(words)

    # Words Per Minute
    wpm = (word_count / duration_seconds) * 60 if duration_seconds > 0 else 0

    return round(wpm, 2), round(duration_seconds, 2)


# ----------------------------
# Fluency Level Mapping
# ----------------------------
def fluency_level(wpm):
    if wpm < 60:
        return "Very Slow"
    elif 60 <= wpm < 90:
        return "Slow"
    elif 90 <= wpm < 120:
        return "Average"
    elif 120 <= wpm < 150:
        return "Good"
    else:
        return "Excellent"


# ----------------------------
# MAIN (Test)
# ----------------------------
if __name__ == "__main__":

    audio_file = "cv-corpus-24.0-2025-12-05/mr/clips/common_voice_mr_37991861.mp3"
    predicted_text = "तिज्कानी गिते ताक कर्मयों क असावर था लावला ही"

    wpm, duration = calculate_fluency(audio_file, predicted_text)
    level = fluency_level(wpm)

    print("\nAUDIO DURATION (seconds):", duration)
    print("WORDS PER MINUTE (WPM):", wpm)
    print("FLUENCY LEVEL:", level)
