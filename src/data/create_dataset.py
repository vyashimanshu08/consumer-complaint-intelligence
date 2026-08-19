from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw_data" / "complaints.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# 2. Settings
# --------------------------------------------------

CHUNK_SIZE = 200_000

SELECTED_COLUMNS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company public response",
    "Company",
    "State",
    "Submitted via",
    "Date sent to company",
    "Company response to consumer",
    "Timely response?",
    "Complaint ID"
]


# --------------------------------------------------
# 3. Create output folder
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 4. Process dataset chunk by chunk
# --------------------------------------------------

chunk_number = 0
total_rows = 0

for chunk in pd.read_csv(
    RAW_FILE,
    usecols=SELECTED_COLUMNS,
    chunksize=CHUNK_SIZE,
    low_memory=False
):

    chunk_number += 1

    # Convert date columns
    chunk["Date received"] = pd.to_datetime(
        chunk["Date received"],
        errors="coerce"
    )

    chunk["Date sent to company"] = pd.to_datetime(
        chunk["Date sent to company"],
        errors="coerce"
    )

    # Remove completely duplicated rows
    chunk = chunk.drop_duplicates()

    # Create missing-value indicators
    chunk["has_narrative"] = (
        chunk["Consumer complaint narrative"].notna()
    )

    chunk["has_public_response"] = (
        chunk["Company public response"].notna()
    )

    # Save each chunk separately
    output_file = OUTPUT_DIR / f"complaints_part_{chunk_number:04d}.csv"

    chunk.to_csv(
        output_file,
        index=False
    )

    total_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows saved: {total_rows:,}"
    )


print("\nProcessing completed successfully!")
print(f"Total chunks: {chunk_number}")
print(f"Total rows saved: {total_rows:,}")