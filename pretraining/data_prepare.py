#
# FINAL, ROBUST SCRIPT for creating long-context fine-tuning datasets.
#
# This script streams data from the Dolma dataset, shuffles it on the fly
# using a buffer, and writes directly to train and validation files, ensuring
# a truly random and well-mixed sample.
#
# You will need to install the required libraries first:
# pip install datasets tiktoken tqdm zstandard
#

import os
import random
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

# --- Configuration Section ---

# 1. Define the single, pre-mixed dataset to use.
DATASET_PATH = "allenai/dolma"
DATASET_CONFIG = "v1_7"
TEXT_COLUMN = "text"

# 2. Set the filtering, splitting, and total size parameters.
TOTAL_DOCS_TO_WRITE = 110000
MIN_TOKENS = 8192
VAL_SPLIT_RATIO = 0.1
OUTPUT_DIR = "data_pretrain"

# 3. Use a fast tokenizer for counting tokens.
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- End of Configuration ---


def main():
    """Main function to process the dataset and create splits."""

    print("Starting dataset creation process...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset stream from {DATASET_PATH} (config: {DATASET_CONFIG})...")

    ds = load_dataset(
        path=DATASET_PATH,
        name=DATASET_CONFIG,
        split="train",
        streaming=True
    )

    #ds = ds.shuffle(buffer_size=10000, seed=42)

    train_filepath = os.path.join(OUTPUT_DIR, "train.txt")
    val_filepath = os.path.join(OUTPUT_DIR, "val.txt")

    print(f"\nWriting up to {TOTAL_DOCS_TO_WRITE} documents with >{MIN_TOKENS} tokens...")

    with open(train_filepath, "w", encoding="utf-8") as f_train, \
         open(val_filepath, "w", encoding="utf-8") as f_val:

        train_docs_written = 0
        val_docs_written = 0

        pbar = tqdm(total=TOTAL_DOCS_TO_WRITE, desc="Writing documents")

        for doc in ds:
            if (train_docs_written + val_docs_written) >= TOTAL_DOCS_TO_WRITE:
                break

            text = doc.get(TEXT_COLUMN)
            if not text or not isinstance(text, str):
                continue

            try:
                tokens = tokenizer.encode(text)
            except Exception:
                continue

            if len(tokens) < MIN_TOKENS:
                continue

            cleaned_text = text.strip().replace("\n", " ")

            if random.random() > VAL_SPLIT_RATIO:
                f_train.write(cleaned_text + "\n")
                train_docs_written += 1
            else:
                f_val.write(cleaned_text + "\n")
                val_docs_written += 1

            pbar.update(1)

        pbar.close()

    print("\n--- Dataset creation complete! ---")
    print(f"Total documents written: {train_docs_written + val_docs_written}")
    print(f"  - Wrote {train_docs_written} documents to: {train_filepath}")
    print(f"  - Wrote {val_docs_written} documents to: {val_filepath}")
    print("------------------------------------")

if __name__ == "__main__":
    main()