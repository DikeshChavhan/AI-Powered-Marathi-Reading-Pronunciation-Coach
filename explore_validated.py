import pandas as pd

DATA_PATH = "cv-corpus-24.0-2025-12-05/mr/validated.tsv"

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    engine="python",        # IMPORTANT
    on_bad_lines="skip"     # Skip corrupted rows
)

print("Columns:", df.columns.tolist())
print("Total validated samples:", len(df))
print(df.head())
