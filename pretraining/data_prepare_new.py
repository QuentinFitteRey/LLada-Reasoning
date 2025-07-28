#
# FINAL, ROBUST SCRIPT for creating long-context fine-tuning datasets with stratified sampling.
#
# This script streams data from the Dolma dataset, shuffles it on the fly
# using a buffer, and then stratifies the selected documents by length
# (e.g., 90% long, 10% short) before writing to train and validation files.
# This ensures a truly random, well-mixed, and length-balanced sample.
#
# You will need to install the required libraries first:
# pip install datasets tiktoken tqdm zstandard
#

import os
import random
from datasets import load_dataset, DownloadConfig
import tiktoken
from tqdm import tqdm

# --- Configuration Section ---

# 1. Define the single, pre-mixed dataset to use.
DATASET_PATH = "allenai/dolma"
DATASET_CONFIG = "v1_7" # v1_7 is a good, diverse mix
TEXT_COLUMN = "text"

# 2. Set the overall target size and splitting parameters.
TOTAL_DOCS_TO_WRITE = 60000 # Your target number of documents
VAL_SPLIT_RATIO = 0.1
OUTPUT_DIR = "data_pretrain_stratified" # Changed output directory to distinguish
MAX_TOKENS_TRUNCATION = 10000 # Maximum tokens for any document, if it's too long

# --- NEW/MODIFIED: Stratified Sampling Parameters ---
# Define the length thresholds and desired ratios for sampling.
LONG_DOC_MIN_TOKENS_THRESHOLD = 8096 # Documents >= this will be considered "long"
TARGET_LONG_DOC_RATIO = 0.90       # 90% of your final dataset will be long docs
TARGET_SHORT_DOC_RATIO = 0.10      # 10% of your final dataset will be short docs
# Ensure ratios sum to 1.0 (or close, due to floating point precision)
assert abs(TARGET_LONG_DOC_RATIO + TARGET_SHORT_DOC_RATIO - 1.0) < 1e-6, \
    "Target long and short doc ratios must sum to 1.0"

# --- NEW/MODIFIED: Data Quality and Overall Length Filtering Parameters ---
# Documents of any length must meet this minimum token count.
MIN_TOKENS_OVERALL = 128 # Minimum token length for any selected document

# Dolma provides 'quality_warnings' and 'perplexity' in its metadata.
# Use these to filter for higher quality.
QUALITY_FILTER_ENABLED = True
PERPLEXITY_THRESHOLD = 150 # Documents with perplexity above this might be noisy/less coherent

# List of quality warnings to exclude (can be customized further).
QUALITY_WARNINGS_TO_EXCLUDE = [
    "boilerplate", "short_line", "long_line", "many_short_lines",
    "bad_encoding", "bad_html", "low_quality", "duplicate",
    "hash_collision", "uncommon_language", "too_few_words", "too_many_words",
    "too_much_boilerplate", # "too_much_code" - include or exclude depending on your needs
    "too_much_non_text", "too_much_repetition", "too_much_symbols", "too_much_whitespace",
    "bad_lang_id", "bad_char_ratio", "bad_word_ratio", # ... (rest of your list, truncated for brevity)
]
# For the full list of QUALITY_WARNINGS_TO_EXCLUDE, refer to the previous script
# and include all the ones you desire to filter out.

# 3. Use a fast tokenizer for counting tokens.
tokenizer = tiktoken.get_encoding("cl100k_base")

# --- End of Configuration ---

def is_high_quality(doc, tokenizer_obj, min_tokens_overall, perplexity_threshold, warnings_to_exclude):
    """
    Checks if a document meets the quality criteria (excluding specific length bands here).
    Length filtering for stratification will happen separately.
    """
    text = doc.get(TEXT_COLUMN)
    if not text or not isinstance(text, str):
        return False

    try:
        tokens = tokenizer_obj.encode(text)
    except Exception:
        return False

    num_tokens = len(tokens)
    if num_tokens < min_tokens_overall: # Apply overall min length
        return False

    # Check Dolma's built-in quality signals
    quality_warnings = doc.get("quality_warnings", [])
    if any(warning in warnings_to_exclude for warning in quality_warnings):
        return False

    # Perplexity score (lower is better)
    perplexity = doc.get("perplexity")
    if perplexity is not None and perplexity > perplexity_threshold:
        return False
    
    return True

def main():
    """Main function to process the dataset and create stratified splits."""

    print("Starting dataset creation process with stratified sampling...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset stream from {DATASET_PATH} (config: {DATASET_CONFIG})...")

    dc = DownloadConfig(
        extract_compressed_file=True,   # auto‑extract any .gz/.zip
        force_download=True,            # don’t re‑download if already cached
    )
    ds = load_dataset(
        path=DATASET_PATH,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
        trust_remote_code=True,        # allow Dolma’s custom loader to run
        download_config=dc            # use our custom download config
    )

    # Enable shuffling for better diversity within the stream buffer
    ds = ds.shuffle(buffer_size=10000, seed=42) # A good buffer size is crucial here

    train_filepath = os.path.join(OUTPUT_DIR, "train_new.txt")
    val_filepath = os.path.join(OUTPUT_DIR, "val_new.txt")

    # Calculate exact numbers of documents needed for each category
    num_long_docs_needed = int(TOTAL_DOCS_TO_WRITE * TARGET_LONG_DOC_RATIO)
    num_short_docs_needed = TOTAL_DOCS_TO_WRITE - num_long_docs_needed # Ensures exact total

    print(f"\nTargeting {TOTAL_DOCS_TO_WRITE} documents:")
    print(f"  - {num_long_docs_needed} long documents (>= {LONG_DOC_MIN_TOKENS_THRESHOLD} tokens)")
    print(f"  - {num_short_docs_needed} short documents (< {LONG_DOC_MIN_TOKENS_THRESHOLD} tokens, >= {MIN_TOKENS_OVERALL} tokens)")
    print(f"Applying quality filters and splitting {VAL_SPLIT_RATIO*100}% for validation...")

    long_docs_buffer = []
    short_docs_buffer = []
    docs_scanned = 0

    # We use a combined progress bar that reflects the total selected documents
    pbar_collect = tqdm(total=TOTAL_DOCS_TO_WRITE, desc="Collecting documents for stratification")

    for doc in ds:
        docs_scanned += 1

        # Stop if we have collected enough documents in both categories
        if len(long_docs_buffer) >= num_long_docs_needed and \
           len(short_docs_buffer) >= num_short_docs_needed:
            pbar_collect.close()
            print(f"Collected {len(long_docs_buffer)} long docs and {len(short_docs_buffer)} short docs.")
            break

        # Apply initial quality filter (common to all documents)
        if QUALITY_FILTER_ENABLED and not is_high_quality(
            doc, tokenizer, MIN_TOKENS_OVERALL, PERPLEXITY_THRESHOLD, QUALITY_WARNINGS_TO_EXCLUDE
        ):
            continue

        # Get token count for length-based stratification
        try:
            tokens = tokenizer.encode(doc.get(TEXT_COLUMN))
            num_tokens = len(tokens)
        except Exception:
            continue # Skip if tokenization fails
        if num_tokens > MAX_TOKENS_TRUNCATION:
 # If the document is too long, truncate tokens and decode back to text
            truncated_tokens = tokens[:MAX_TOKENS_TRUNCATION]
            truncated_text = tokenizer.decode(truncated_tokens)
# Update the document's text and token count for subsequent processing
            doc[TEXT_COLUMN] = truncated_text
            num_tokens = len(truncated_tokens) 
        # Classify and add to appropriate buffer if still needed
        if num_tokens >= LONG_DOC_MIN_TOKENS_THRESHOLD and len(long_docs_buffer) < num_long_docs_needed:
            long_docs_buffer.append(doc)
            pbar_collect.update(1)
        elif num_tokens < LONG_DOC_MIN_TOKENS_THRESHOLD and len(short_docs_buffer) < num_short_docs_needed:
            # Also ensure short docs meet the overall minimum token requirement
            if num_tokens >= MIN_TOKENS_OVERALL:
                short_docs_buffer.append(doc)
                pbar_collect.update(1)
        
        # Optional: Print progress on scanned documents if collection is slow
        if docs_scanned % 100000 == 0: # Print every 100k docs scanned
            print(f"Scanned {docs_scanned} documents. Current buffer sizes: "
                  f"Long={len(long_docs_buffer)}/{num_long_docs_needed}, "
                  f"Short={len(short_docs_buffer)}/{num_short_docs_needed}")


    pbar_collect.close() # Ensure pbar is closed if loop finishes early due to break

    # Combine and shuffle the collected documents for final train/val split
    all_selected_docs = long_docs_buffer + short_docs_buffer
    random.shuffle(all_selected_docs) # Shuffle the combined list one more time

    print(f"\nFinal count of collected documents: {len(all_selected_docs)}")
    if len(all_selected_docs) < TOTAL_DOCS_TO_WRITE:
        print(f"Warning: Could only collect {len(all_selected_docs)} out of {TOTAL_DOCS_TO_WRITE} target documents. "
              "Consider increasing TOTAL_DOCS_TO_WRITE, reducing MIN_TOKENS_OVERALL, or adjusting filters if the dataset is exhausted.")

    train_docs_written = 0
    val_docs_written = 0

    with open(train_filepath, "w", encoding="utf-8") as f_train, \
         open(val_filepath, "w", encoding="utf-8") as f_val:

        pbar_write = tqdm(total=len(all_selected_docs), desc="Writing train/val files")

        for doc in all_selected_docs:
            text_content = doc.get(TEXT_COLUMN)
            cleaned_text = text_content.strip().replace("\n", " ")

            if random.random() > VAL_SPLIT_RATIO:
                f_train.write(cleaned_text + "\n")
                train_docs_written += 1
            else:
                f_val.write(cleaned_text + "\n")
                val_docs_written += 1
            pbar_write.update(1)
        
        pbar_write.close()

    print("\n--- Dataset creation complete! ---")
    print(f"Total documents written: {train_docs_written + val_docs_written}")
    print(f"  - Wrote {train_docs_written} documents to: {train_filepath}")
    print(f"  - Wrote {val_docs_written} documents to: {val_filepath}")
    print("------------------------------------")

if __name__ == "__main__":
    main()