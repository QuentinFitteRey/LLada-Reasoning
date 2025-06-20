#
# Efficient script for creating long-context fine-tuning datasets.
#
# This script streams data from Hugging Face, processes it on the fly,
# and writes directly to train and validation files. It's memory-efficient
# and suitable for very large datasets.
#
# You will need to install the required libraries first:
# pip install datasets tiktoken tqdm
#

import os
import random
from datasets import load_dataset
import tiktoken
from tqdm import tqdm

DATA_CONFIG = {
        "SlimPajama-ArXiv": {
        "path": "cerebras/SlimPajama-627B",
        "name": "default",
        "split": "train",
        "max_docs": 10000,
        "text_col": "text"
    },
    "SlimPajama-Books": {
        "path": "cerebras/SlimPajama-627B",
        "name": "default",
        "split": "train",
        "max_docs": 5000, # Take 5,000 books
        "text_col": "text"
    },
    "The-Stack-v2-Python": {
        "path": "bigcode/the-stack-v2",
        "name": "python", 
        "split": "train",
        "max_docs": 10000, 
        "text_col": "content"
    }
}

MIN_TOKENS = 8192    
VAL_SPLIT_RATIO = 0.01
OUTPUT_DIR = "data"

tokenizer = tiktoken.get_encoding("cl100k_base")



def main():
    """Main function to process datasets and create splits."""
    
    print("Starting dataset creation process...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_filepath = os.path.join(OUTPUT_DIR, "train.txt")
    val_filepath = os.path.join(OUTPUT_DIR, "val.txt")

    # Open the final output files once
    with open(train_filepath, "w", encoding="utf-8") as f_train, \
         open(val_filepath, "w", encoding="utf-8") as f_val:

        total_docs_written = 0
        train_docs_written = 0
        val_docs_written = 0

        # Process each dataset defined in the config
        for name, config in DATA_CONFIG.items():
            print(f"\n---> Processing dataset: {name}")

            ds = load_dataset(
                path=config["path"],
                name=config.get("name"), # Use .get() for optional keys
                split=config["split"],
                streaming=True  # The magic flag for efficiency!
            )

            docs_processed = 0
            # Use tqdm for a progress bar
            for doc in tqdm(ds, desc=f"Streaming {name}"):
                # Stop if we have processed enough documents from this source
                if docs_processed >= config["max_docs"]:
                    break
                
                text = doc.get(config["text_col"])
                if not text or not isinstance(text, str):
                    continue # Skip if the text column is missing or not a string

                # Filter by token count
                # This is more accurate than character count for LLMs
                tokens = tokenizer.encode(text)
                if len(tokens) < MIN_TOKENS:
                    continue

                # Clean the text: strip whitespace and replace newlines to keep one doc per line
                cleaned_text = text.strip().replace("\n", " ")

                # Randomly assign to train or validation split
                if random.random() > VAL_SPLIT_RATIO:
                    f_train.write(cleaned_text + "\n")
                    train_docs_written += 1
                else:
                    f_val.write(cleaned_text + "\n")
                    val_docs_written += 1
                
                total_docs_written += 1
                docs_processed += 1

    print("\n--- Dataset creation complete! ---")
    print(f"Total documents processed from sources: {sum(cfg['max_docs'] for cfg in DATA_CONFIG.values())}")
    print(f"Total documents that met the minimum token count ({MIN_TOKENS}): {total_docs_written}")
    print(f"  - Wrote {train_docs_written} documents to: {train_filepath}")
    print(f"  - Wrote {val_docs_written} documents to: {val_filepath}")
    print("------------------------------------")

if __name__ == "__main__":
    main()