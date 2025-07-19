#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer

# --- Configuration ---
SOURCE_DATASET_ID = "nvidia/Llama-Nemotron-Post-Training-Dataset"
CATEGORIES_TO_SAMPLE = {
    "math": 50000,
    "code": 50000,
    "science": 50000,
    "instruction_following": 50000,
    "chat": 50000,
    "safety": 50000,
}
AVG_CHARS_PER_TOKEN = 4

# --- Data Conversion & Processing Functions ---

def to_conversation(example):
    """
    ## --- MODIFIED ---
    Converts a Nemotron example to the user/assistant format,
    prepending the 'system_prompt' column from the dataset to the user's message.
    """
    user_content = example.get("input", "")
    # Get the system prompt from the example itself
    system_prompt = example.get("system_prompt", "")

    # Combine the system prompt and user content into a single user message
    if system_prompt:
        full_user_prompt = system_prompt + " " + user_content
    else:
        print("Warning: No system prompt found in example, using only user content. PAS NORMAL")
        full_user_prompt = user_content
        
    return {"user": full_user_prompt, "assistant": example.get("output", "")}

def preprocess_batch(examples, tokenizer, max_len):
    """
    Processes a batch of examples to filter by length using simple concatenation.
    """
    char_limit = AVG_CHARS_PER_TOKEN * max_len
    
    convs = []
    num_examples = len(examples[next(iter(examples))])
    for i in range(num_examples):
        rec = {col: examples[col][i] for col in examples}
        c = to_conversation(rec)
        if len(c["user"]) + len(c["assistant"]) <= char_limit:
            convs.append(c)

    texts = [c["user"] + tokenizer.eos_token + c["assistant"] for c in convs]
    tokenized_outputs = tokenizer(texts, add_special_tokens=False, truncation=False)

    final_convs = [
        convs[i] for i, ids in enumerate(tokenized_outputs["input_ids"]) if len(ids) <= max_len
    ]

    return {
        "user": [c["user"] for c in final_convs],
        "assistant": [c["assistant"] for c in final_convs],
    }

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Filter and sample the Llama-Nemotron dataset by category and length."
    )
    parser.add_argument("--model_name", type=str, default="./llada_local", help="Tokenizer model to use for length calculation.")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum context length in tokens.")
    parser.add_argument("--out_dir", type=str, default="./nemotron_sft_sample", help="Directory to save the final processed dataset.")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of processes for dataset mapping.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for preprocessing.")
    args = parser.parse_args()

    print(f"🔵 Loading tokenizer: '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    print(f"🔵 Loading the full dataset: {SOURCE_DATASET_ID}...")
    full_dataset = load_dataset(SOURCE_DATASET_ID, split="train")
    print(f"✅ Full dataset loaded with {len(full_dataset):,} examples.")

    train_subsets = []
    val_subsets = []
    category_stats = {}

    for category, sample_size in CATEGORIES_TO_SAMPLE.items():
        print(f"\n🔎 Processing category: '{category}'")

        filtered_by_category = full_dataset.filter(
            lambda example: example["category"] == category,
            num_proc=args.num_proc
        )

        print(f"Filtering for max length ({args.max_length} tokens)...")
        valid_length_subset = filtered_by_category.map(
            lambda batch: preprocess_batch(batch, tokenizer, args.max_length),
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            remove_columns=filtered_by_category.column_names,
        )

        available_examples = len(valid_length_subset)
        print(f"📊 Found {available_examples:,} valid examples for '{category}' after length filtering.")
        
        if available_examples < sample_size:
            print(f"⚠️ WARNING: Using all available {available_examples:,} examples, as this is less than the requested {sample_size:,}.")
            final_sample_size = available_examples
        else:
            final_sample_size = sample_size

        if final_sample_size < 20: # Need at least a few samples to create a split
            print(f"SKIPPING '{category}' as not enough samples ({final_sample_size}) are available to create a validation split.")
            category_stats[category] = {"train": 0, "val": 0}
            continue
            
        print(f"Shuffling and selecting {final_sample_size:,} random examples...")
        final_sample = valid_length_subset.shuffle(seed=42).select(range(final_sample_size))

        print(f"Splitting '{category}' sample into 95% train and 5% validation...")
        split_dataset = final_sample.train_test_split(test_size=0.05, seed=42)

        train_subsets.append(split_dataset["train"])
        val_subsets.append(split_dataset["test"])
        
        category_stats[category] = {
            "train": len(split_dataset["train"]),
            "val": len(split_dataset["test"]),
        }

    if not train_subsets:
        print("\n❌ No data was sampled. Exiting.")
        return
        
    print("\n" + "="*60)
    print("📊 Final Sample Statistics:")
    total_train, total_val = 0, 0
    print(f"  - {'CATEGORY':<22} | {'TRAIN':>10} | {'VALIDATION':>12}")
    print("  " + "-"*50)
    for category, counts in category_stats.items():
        print(f"  - {category:<22} | {counts['train']:>10,} | {counts['val']:>12,}")
        total_train += counts['train']
        total_val += counts['val']
    print("  " + "-"*50)
    print(f"  - {'TOTAL':<22} | {total_train:>10,} | {total_val:>12,}")
    print("="*60)
    
    print("\n🔗 Combining all train and validation subsets...")
    combined_train_ds = concatenate_datasets(train_subsets).shuffle(seed=42)
    combined_val_ds = concatenate_datasets(val_subsets).shuffle(seed=42)
    
    print(f"\n💾 Saving final dataset...")
    output_dd = DatasetDict({
        "train": combined_train_ds,
        "validation": combined_val_ds
    })
    os.makedirs(args.out_dir, exist_ok=True)
    output_dd.save_to_disk(args.out_dir)
    print(f"\n✅ Success! Dataset saved to '{args.out_dir}'")
    print(f"Final train samples: {len(combined_train_ds):,}")
    print(f"Final validation samples: {len(combined_val_ds):,}")

if __name__ == "__main__":
    main()