import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw_data/complaints.csv")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

CHUNK_SIZE = 50000
total_rows = 0
chunk_number = 0

output_file = PROCESSED_DIR / "complaints_processed_test.csv"

for chunk in pd.read_csv(
    RAW_FILE,
    usecols=SELECTED_COLUMNS,
    chunksize=CHUNK_SIZE
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

    # Handle missing categorical values
    categorical_missing = [
        "Sub-product",
        "Sub-issue",
        "State"
    ]

    for col in categorical_missing:
        chunk[col] = chunk[col].fillna("Unknown")

    # Create missingness indicators
    chunk["has_narrative"] = (
        chunk["Consumer complaint narrative"].notna().astype(int)
    )

    chunk["has_public_response"] = (
        chunk["Company public response"].notna().astype(int)
    )

    # Remove completely duplicated rows
    chunk = chunk.drop_duplicates()

    # Save processed chunk
    chunk.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not output_file.exists()
    )

    total_rows += len(chunk)

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows saved: {total_rows:,}"
    )

    if chunk_number == 5:
        break

print(f"Test complete. Rows saved: {total_rows:,}")