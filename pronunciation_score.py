import re
from jiwer import wer
from difflib import SequenceMatcher

# ----------------------------
# Text normalization (Marathi-safe)
# ----------------------------
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)  # keep Devanagari only
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------
# Word-level difference detection
# ----------------------------
def get_word_differences(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()

    matcher = SequenceMatcher(None, ref_words, hyp_words)
    mistakes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            ref_chunk = " ".join(ref_words[i1:i2])
            hyp_chunk = " ".join(hyp_words[j1:j2])
            mistakes.append((ref_chunk, hyp_chunk))

    return mistakes


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    expected_text = "टिळकांनी गीतेचा कर्मयोग असा अर्थ लावला आहे."
    predicted_text = "तिज्कानी गिते ताक कर्मयों क असावर था लावला ही"

    # Normalize texts
    expected_norm = normalize_text(expected_text)
    predicted_norm = normalize_text(predicted_text)

    # ----------------------------
    # Word Error Rate (WER)
    # ----------------------------
    error_rate = wer(expected_norm, predicted_norm)

    # Convert to user-friendly accuracy (clamped)
    raw_accuracy = (1 - error_rate) * 100
    accuracy = max(0, min(100, raw_accuracy))

    # Detect pronunciation mistakes
    mistakes = get_word_differences(expected_norm, predicted_norm)

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print("\nEXPECTED (Normalized):")
    print(expected_norm)

    print("\nPREDICTED (Normalized):")
    print(predicted_norm)

    print("\nPRONUNCIATION ACCURACY:")
    print(f"{accuracy:.2f}%")

    print("\nMISPRONOUNCED / MISMATCHED WORDS:")
    if mistakes:
        for ref, hyp in mistakes:
            print(f"- Expected: '{ref}'  →  Spoken: '{hyp}'")
    else:
        print("No major pronunciation mistakes detected 🎉")
