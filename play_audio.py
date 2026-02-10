import pandas as pd
import os

DATA_PATH = "cv-corpus-24.0-2025-12-05/mr/validated.tsv"
CLIPS_PATH = "cv-corpus-24.0-2025-12-05/mr/clips"

df = pd.read_csv(DATA_PATH, sep="\t", engine="python", on_bad_lines="skip")

row = df.iloc[0]

audio_path = os.path.join(CLIPS_PATH, row["path"])

print("Sentence (Expected):")
print(row["sentence"])
print("\nAudio file path:")
print(audio_path)
