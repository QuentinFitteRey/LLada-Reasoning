#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Configuration ---
# !!! IMPORTANT !!!
# Please update this with the Hugging Face ID of your dataset.
# Example: "HuggingFaceH4/ultrafeedback_binarized" or "cognitivecomputations/dolphin"
DEFAULT_DATASET_ID = "open-thoughts/OpenThoughts-114k" 

AVG_CHARS_PER_TOKEN = 4  # Heuristic for fast pre-filtering

# --- Data Processing Function ---

def preprocess_batch(examples, tokenizer, max_len):
    """
    Processes a batch of examples, assuming a single user/assistant turn.
    It combines the 'system' prompt with the user's message and filters by length.
    """
    # 1. Fast, heuristic-based pre-filtering on character count
    char_limit = AVG_CHARS_PER_TOKEN * max_len
    
    candidate_convs = []
    num_examples = len(examples[next(iter(examples))])
    
    # Process each example, directly accessing the single user/assistant turn
    for i in range(num_examples):
        conversation_list = examples["conversations"][i]
        
        # Skip any malformed examples that don't fit the expected structure
        if not (len(conversation_list) == 2 and conversation_list[0].get("from") == "user" and conversation_list[1].get("from") == "assistant"):
            continue

        system_prompt = examples.get("system", [""]*num_examples)[i]
        user_message = conversation_list[0].get("value", "")
        assistant_message = conversation_list[1].get("value", "")
        
        # Combine system prompt with the user message. Use strip() to handle empty system prompts gracefully.
        user_text = f"{system_prompt}\n\n{user_message}".strip()
        
        # Heuristic check to quickly discard very long examples
        if len(user_text) + len(assistant_message) <= char_limit:
            candidate_convs.append({"user": user_text, "assistant": assistant_message})

    # 2. Precise filtering based on actual token count
    if not candidate_convs:
        return {"user": [], "assistant": []}
        
    texts_to_tokenize = [c["user"] + tokenizer.eos_token + c["assistant"] for c in candidate_convs]
    tokenized_outputs = tokenizer(texts_to_tokenize, add_special_tokens=False, truncation=False)

    # Keep only conversations that are strictly within the max_len
    final_convs = [
        candidate_convs[i] 
        for i, ids in enumerate(tokenized_outputs["input_ids"]) 
        if len(ids) <= max_len
    ]

    # Return the filtered conversations in the final format
    if not final_convs:
        return {"user": [], "assistant": []}
        
    return {
        "user": [c["user"] for c in final_convs],
        "assistant": [c["assistant"] for c in final_convs],
    }

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Filter a single-turn conversational dataset by length and create train/validation splits."
    )
    parser.add_argument("--hf_dataset_id", type=str, default=DEFAULT_DATASET_ID, help="Hugging Face dataset ID to process.")
    parser.add_argument("--model_name", type=str, default="./llada_local_1.5", help="Tokenizer model path for length calculation.")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum context length in tokens.")
    parser.add_argument("--out_dir", type=str, default="./filtered_conversational_dataset_4k", help="Directory to save the final processed dataset.")
    parser.add_argument("--val_split_size", type=float, default=0.02, help="Proportion of data for validation (e.g., 0.02 for 2%).")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of processes for dataset mapping.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for preprocessing.")
    
    args = parser.parse_args()

    if args.hf_dataset_id == "your/new-dataset-id":
        print("🛑 Error: Please specify your dataset using the --hf_dataset_id argument.")
        return

    print(f"🔵 Loading tokenizer: '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    print(f"🔵 Loading dataset: {args.hf_dataset_id}...")
    # Assuming the dataset has a 'train' split, adjust if necessary
    full_dataset = load_dataset(args.hf_dataset_id, split="train") 
    original_columns = full_dataset.column_names
    print(f"✅ Full dataset loaded with {len(full_dataset):,} examples.")

    print(f"\n🔎 Filtering all examples for max length ({args.max_length} tokens)...")
    filtered_dataset = full_dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, args.max_length),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=original_columns, # Remove original columns ('system', 'conversations', etc.)
    )
    print(f"📊 Found {len(filtered_dataset):,} valid examples after length filtering.")

    if len(filtered_dataset) < 2:
        print("\n❌ Not enough data after filtering to create a train/validation split. Exiting.")
        return

    print(f"\n🔀 Shuffling and splitting dataset ({1-args.val_split_size:.0%} train, {args.val_split_size:.0%} val)...")
    final_dataset_splits = filtered_dataset.train_test_split(test_size=args.val_split_size, seed=42)

    print(f"\n💾 Saving final dataset to '{args.out_dir}'...")
    os.makedirs(args.out_dir, exist_ok=True)
    final_dataset_splits.save_to_disk(args.out_dir)
    
    print("\n" + "="*50)
    print("✅ Success! Dataset saved.")
    print(f"Final train samples:      {len(final_dataset_splits['train']):,}")
    print(f"Final validation samples: {len(final_dataset_splits['test']):,}")
    print("="*50)

if __name__ == "__main__":
    main()