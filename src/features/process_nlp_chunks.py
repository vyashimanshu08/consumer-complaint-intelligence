# ============================================================
# NLP CHUNK PROCESSING
# Consumer Complaint Intelligence Project
# ============================================================


# Step 1: Import necessary libraries and project root

import pandas as pd
from pathlib import Path
import sys


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing from src/features
sys.path.append(str(PROJECT_ROOT / "src" / "features"))

from text_preprocessing import clean_narrative


# ============================================================
# Step 2: Define input and output directories
# ============================================================

# Input directory containing processed complaint chunks
INPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Output directory for NLP-processed chunks
OUTPUT_DIR = PROJECT_ROOT / "data" / "nlp_processed"

# Create output directory if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 3: Define chunk range
# ============================================================

# Chunk 0001 has already been processed separately
START_CHUNK = 3

# For testing, process only chunk 0002 first
END_CHUNK = 86


# ============================================================
# Step 4: Process each chunk
# ============================================================

for chunk_number in range(START_CHUNK, END_CHUNK + 1):

    input_file = INPUT_DIR / f"complaints_part_{chunk_number:04d}.csv"

    output_file = OUTPUT_DIR / f"complaints_nlp_part_{chunk_number:04d}.csv"


    print("\n" + "=" * 60)
    print(f"Processing chunk {chunk_number:04d}")
    print(f"Input : {input_file.name}")
    print(f"Output: {output_file.name}")
    print("=" * 60)


    # --------------------------------------------------------
    # Check whether input file exists
    # --------------------------------------------------------

    if not input_file.exists():

        print(f"ERROR: Input file not found: {input_file}")
        continue


    # --------------------------------------------------------
    # Read chunk
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    print(f"Rows loaded: {len(df):,}")


    # --------------------------------------------------------
    # Select rows containing narratives
    # --------------------------------------------------------

    narrative_mask = df["has_narrative"] == True

    narratives = df.loc[
        narrative_mask,
        "Consumer complaint narrative"
    ].dropna()

    print(f"Narratives found: {len(narratives):,}")


    # --------------------------------------------------------
    # Apply NLP preprocessing
    # --------------------------------------------------------

    print("Applying NLP preprocessing...")

    cleaned_narratives = narratives.apply(clean_narrative)

    print("NLP preprocessing completed.")


    # --------------------------------------------------------
    # Add cleaned narratives back to dataframe
    # --------------------------------------------------------

    df.loc[
        cleaned_narratives.index,
        "clean_narrative"
    ] = cleaned_narratives


    # --------------------------------------------------------
    # Save NLP-processed chunk
    # --------------------------------------------------------

    df.to_csv(output_file, index=False)

    print(f"Saved successfully: {output_file.name}")
    print(f"Rows saved: {len(df):,}")


# ============================================================
# Step 5: Processing completed
# ============================================================

print("\n" + "=" * 60)
print("NLP CHUNK PROCESSING COMPLETED")
print("=" * 60)