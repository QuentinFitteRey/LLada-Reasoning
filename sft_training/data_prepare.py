#!/usr/bin/env python3
import argparse
import itertools
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm.auto import tqdm

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
        sys_p = example.get("system", "").strip()
        users, assists = [], []
        for turn in example.get("conversations", []):
            who = turn.get("from", "").lower()
            txt = turn.get("value", "").strip()
            if who in ("user", "human"):
                users.append(txt)
            elif who not in ("system",):
                assists.append(txt)
        user_str = (sys_p + "\n\n") if sys_p else ""
        user_str += " ".join(users)
        return {"user": user_str, "assistant": " ".join(assists)}
    # Fallback
    joined = "\n".join(f"{k}: {v}" for k, v in example.items())
    return {"user": joined, "assistant": ""}


def process_stream(ds_id: str, tokenizer, max_length: int, char_limit: int):
    """
    Stream through all splits, apply filters, and collect up to MAX_EXAMPLES.
    """
    collected = []
    # Attempt to stream SFT subset
    try:
        ds_stream = load_dataset(ds_id, "SFT", streaming=True)
        print(f"→ Streaming SFT splits: {list(ds_stream.keys())}")
    except Exception:
        ds_stream = load_dataset(ds_id, streaming=True)
        print(f"→ Streaming default splits: {list(ds_stream.keys())}")

    # Chain all split generators
    iterators = [ds_stream[k] for k in ds_stream]
    flat_iter = itertools.chain.from_iterable(iterators)
    total = MAX_DEBUG_EXAMPLES if DEBUG else MAX_EXAMPLES

    for ex in tqdm(flat_iter, desc=f"Collecting {ds_id}", total=total):
        conv = to_conversation(ex, ds_id)
        # quick char filter
        if len(conv["user"]) + len(conv["assistant"]) > char_limit:
            continue
        # token filter
        ids = tokenizer(conv["user"] + tokenizer.eos_token + conv["assistant"], add_special_tokens=False)["input_ids"]
        if len(ids) > max_length:
            continue
        collected.append({"user": conv["user"], "assistant": conv["assistant"]})
        # stop early once we have enough
        if len(collected) >= total:
            break

    print(f"→ Collected {len(collected)} examples from {ds_id}")
    # Build dataset and split
    ds = Dataset.from_list(collected)
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=0.2, seed=42)
    return split


def prepare_data(output_path: str, model_name: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    avg_chars_per_token = 4
    char_limit = avg_chars_per_token * max_length
    all_splits = {}

    for ds_id in DATASETS:
        print(f"\n🔍 Processing dataset: {ds_id}")
        split = process_stream(ds_id, tokenizer, max_length, char_limit)
        all_splits[f"{ds_id.split('/')[-1]}_train"] = split["train"]
        all_splits[f"{ds_id.split('/')[-1]}_validation"] = split["test"]

    combined = DatasetDict(all_splits)
    combined.save_to_disk(output_path)
    print(f"\n✅ Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stream & filter SFT data for SFT training")
    parser.add_argument("--model_name", type=str, default="../llada_local_instruct")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--out", type=str, default="./sft_data")
    args = parser.parse_args()

    print(f"Preparing SFT data into {args.out} with max {args.max_length} tokens…")
    prepare_data(args.out, args.model_name, args.max_length)
    print("Done.")