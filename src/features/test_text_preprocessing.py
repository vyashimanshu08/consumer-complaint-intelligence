import pandas as pd
from pathlib import Path
import sys


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing from src/features
sys.path.append(str(PROJECT_ROOT / "src" / "features"))

from text_preprocessing import clean_narrative


# Input file
FILE_PATH = PROJECT_ROOT / "data" / "processed" / "complaints_part_0001.csv"


# NLP output folder
OUTPUT_DIR = PROJECT_ROOT / "data" / "nlp_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file
OUTPUT_FILE = OUTPUT_DIR / "complaints_nlp_part_0001.csv"
df = pd.read_csv(FILE_PATH)

print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

# Narrative avable in the dataset
# Check narrative availability
print("\nNarrative availability:")
print(df["has_narrative"].value_counts())

# Select only rows containing narratives
narratives = df.loc[
    df["has_narrative"] == True,
    "Consumer complaint narrative"
].dropna()

print("\nNumber of narratives:", len(narratives))


# Apply NLP preprocessing
cleaned_narratives = narratives.apply(clean_narrative)

print("\nPreprocessing completed.")
print("Cleaned narratives:", len(cleaned_narratives))


# Compare original vs cleaned text
for i in range(5):
    print(f"\n--- Narrative {i + 1} ---")

    print("Original:")
    print(narratives.iloc[i][:500])

    print("\nCleaned:")
    print(cleaned_narratives.iloc[i][:500])


# Validation checks
print("\n========== VALIDATION ==========")

# 1. Check empty cleaned narratives
empty_cleaned = (cleaned_narratives.str.strip() == "").sum()

print("Empty cleaned narratives:", empty_cleaned)

# 2. Check redaction tokens
remaining_redactions = cleaned_narratives.str.findall(
    r"\bx{2,}\b"
).explode().dropna()

print("Remaining redaction tokens:", len(remaining_redactions))

# 3. Check preserved negations
negation_words = {
    "not", "no", "never", "neither",
    "nor", "without", "against",
    "before", "after", "under",
    "over", "between", "during"
}

negation_count = sum(
    any(word in negation_words for word in text.split())
    for text in cleaned_narratives
)

print("Narratives containing preserved negation words:", negation_count)

# Add cleaned narratives back to the dataframe
df.loc[narratives.index, "clean_narrative"] = cleaned_narratives

# Save NLP-processed chunk
df.to_csv(OUTPUT_FILE, index=False)

print("\nNLP-processed file saved successfully.")
print("Output:", OUTPUT_FILE)
print("Rows saved:", len(df))