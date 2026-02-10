import os
import pandas as pd
import whisper

# ----------------------------
# Paths (update ONLY if folder name changes)
# ----------------------------
DATA_PATH = "cv-corpus-24.0-2025-12-05/mr/validated.tsv"
CLIPS_PATH = "cv-corpus-24.0-2025-12-05/mr/clips"

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH, sep="\t", engine="python", on_bad_lines="skip")

print(f"Total validated samples: {len(df)}")

# Pick one sample (you can change index later)
row = df.iloc[0]

audio_file = os.path.join(CLIPS_PATH, row["path"])
expected_text = row["sentence"]

# ----------------------------
# Load Whisper model
# ----------------------------
model = whisper.load_model("small")  # you can try "small" later

# ----------------------------
# Transcribe audio (FORCED Marathi)
# ----------------------------
result = model.transcribe(
    audio_file,
    language="mr",
    task="transcribe",
    temperature=0.0,
    beam_size=5,
    best_of=5,
    condition_on_previous_text=False,
    no_speech_threshold=0.2,
    logprob_threshold=-1.0
)


predicted_text = result["text"]

# ----------------------------
# Output
# ----------------------------
print("\nEXPECTED TEXT:")
print(expected_text)

print("\nPREDICTED TEXT:")
print(predicted_text)
