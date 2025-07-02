#!/usr/bin/env python3
import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

# Configuration
DATASETS = [
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "bespokelabs/Bespoke-Stratos-17k",
    "open-thoughts/OpenThoughts-114k",
    "databricks/databricks-dolly-15k",
    "Open-Orca/SlimOrca",
]

DEBUG = False  # Toggle for quick local testing
MAX_DEBUG_EXAMPLES = 500  # For DEBUG mode
MAX_EXAMPLES = 500_000    # Final cap after filtering

NUMPROC = 8

def to_conversation(example, ds_id: str):
    if ds_id.startswith("nvidia/Llama-Nemotron"):
        user = " ".join(msg.get("content", "") for msg in example.get("input", []))
        return {"user": user, "assistant": example.get("output", "")}
    if ds_id == "databricks/databricks-dolly-15k":
        instr = example.get("instruction", "").strip()
        ctx = example.get("context", "").strip()
        prompt = f"{instr}\n\n{ctx}" if ctx else instr
        return {"user": prompt, "assistant": example.get("response", "")}
    if "system" in example and "conversations" in example:
        sys_p = example["system"].strip()
        users, assists = [], []
        for turn in example["conversations"]:
            who = turn.get("from", "").lower()
            txt = turn.get("value", "").strip()
            if who in ("user", "human"):
                users.append(txt)
            elif who not in ("system",):
                assists.append(txt)
        user_str = (sys_p + "\n\n") if sys_p else ""
        user_str += " ".join(users)
        return {"user": user_str, "assistant": " ".join(assists)}
    # fallback
    joined = "\n".join(f"{k}: {v}" for k, v in example.items())
    return {"user": joined, "assistant": ""}

def prepare_user_assistant(output_path: str, model_name: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    avg_chars_per_token = 4
    char_limit = avg_chars_per_token * max_length
    all_splits = {}

    for ds_id in DATASETS:
        print(f"\n🔍 Processing dataset: {ds_id}")
        # Load dataset (try SFT subset)
        try:
            ds_all = load_dataset(ds_id, "SFT")
            print("→ Loaded 'SFT' subset")
        except Exception:
            ds_all = load_dataset(ds_id)
            print("→ Loaded default dataset")

        # Normalize to train/validation[/test]
        if isinstance(ds_all, DatasetDict):
            splits = ds_all
            train_keys = [k for k in splits.keys() if k.lower() not in ("validation", "valid", "test")]
            train_set = concatenate_datasets([splits[k] for k in train_keys])
            if "validation" in splits:
                val_set = splits["validation"]
            else:
                print("→ Creating validation split from train (20%)")
                tmp = train_set.train_test_split(test_size=0.2, seed=42)
                train_set, val_set = tmp["train"], tmp["test"]
            new_splits = {"train": train_set, "validation": val_set}
            if "test" in splits:
                new_splits["test"] = splits["test"]
            builder = DatasetDict(new_splits)
        else:
            print("→ Creating validation split from data (20%)")
            tmp = ds_all.train_test_split(test_size=0.2, seed=42)
            builder = DatasetDict({"train": tmp["train"], "validation": tmp["test"]})

        # Process each split
        for split_name, ds in builder.items():
            print(f"→ Split: {split_name}", end=" … ")

            # DEBUG: take a small subset first
            if DEBUG:
                ds = ds.select(range(min(len(ds), MAX_DEBUG_EXAMPLES)))
                print(f"(DEBUG mode: {len(ds)} examples)", end=" … ")

            # Stage 0: compute token lengths and sort ascending
            def add_length(ex):
                conv = to_conversation(ex, ds_id)
                text = conv["user"] + tokenizer.eos_token + conv["assistant"]
                return {"length": len(tokenizer(text, add_special_tokens=False)["input_ids"])}
            ds = ds.map(add_length, num_proc=NUMPROC)
            print(f"computed token lengths: {len(ds)} examples", end=" … ")
            ds = ds.sort("length")
            ds = ds.remove_columns("length")
            print(f"sorted by token length", end=" … ")

            # Stage 1: quick char-count filter
            def char_ok(ex):
                conv = to_conversation(ex, ds_id)
                return len(conv["user"]) + len(conv["assistant"]) <= char_limit
            ds_char = ds.filter(char_ok, batched=False, num_proc=NUMPROC)
            print(f"after char-filter: {len(ds_char)}/{len(ds)}", end=" … ")

            # Stage 2: normalize to user/assistant and token-length filter
            conv_ds = ds_char.map(
                lambda ex: to_conversation(ex, ds_id),
                remove_columns=ds_char.column_names,
                num_proc=NUMPROC
            )
            def token_ok_batch(examples):
                texts = [u + tokenizer.eos_token + a for u, a in zip(examples["user"], examples["assistant"])]
                toks = tokenizer(texts, add_special_tokens=False)
                return [len(ids) <= max_length for ids in toks["input_ids"]]
            conv_filt = conv_ds.filter(token_ok_batch, batched=True, batch_size=1000, num_proc=NUMPROC)
            print(f"after token-filter: {len(conv_filt)}/{len(conv_ds)}", end=" … ")

            # Stage 3: cap to MAX_EXAMPLES after filtering
            final_ds = conv_filt.select(range(min(len(conv_filt), MAX_EXAMPLES)))
            print(f"capped to {len(final_ds)} examples", end=" … ")

            all_splits[f"{ds_id.split('/')[-1]}_{split_name}"] = final_ds
            print("done.")

    combined = DatasetDict(all_splits)
    combined.save_to_disk(output_path)
    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Prepare & filter SFT data as user/assistant pairs")
    parser.add_argument("--model_name", type=str, default="../llada_local_instruct")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--out", type=str, default="./sft_data")
    args = parser.parse_args()

    print(f"Preparing SFT data into {args.out} with max {args.max_length} tokens…")
    prepare_user_assistant(args.out, args.model_name, args.max_length)
    print("Done.")