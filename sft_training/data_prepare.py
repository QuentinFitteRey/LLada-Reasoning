#!/usr/bin/env python3
import argparse
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

DATASETS = [
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "bespokelabs/Bespoke-Stratos-17k",
    "open-thoughts/OpenThoughts-114k",
    "databricks/databricks-dolly-15k",
    "Open-Orca/SlimOrca",
]

DEBUG = False  # Set to True for quick testing with a small subset

def to_conversation(example, ds_id: str):
    if ds_id.startswith("nvidia/Llama-Nemotron"):
        user = " ".join(msg.get("content","") for msg in example.get("input", []))
        return {"user": user, "assistant": example.get("output","")}
    if ds_id == "databricks/databricks-dolly-15k":
        instr = example.get("instruction","").strip()
        ctx   = example.get("context","").strip()
        prompt = f"{instr}\n\n{ctx}" if ctx else instr
        return {"user": prompt, "assistant": example.get("response","")}
    if "system" in example and "conversations" in example:
        sys_p  = example["system"].strip()
        users, assists = [], []
        for turn in example["conversations"]:
            who = turn.get("from","").lower()
            txt = turn.get("value","").strip()
            if who in ("user","human"):
                users.append(txt)
            elif who not in ("system",):
                assists.append(txt)
        user_str = (sys_p + "\n\n") if sys_p else ""
        user_str += " ".join(users)
        return {"user": user_str, "assistant": " ".join(assists)}
    # fallback
    joined = "\n".join(f"{k}: {v}" for k,v in example.items())
    return {"user": joined, "assistant": ""}

def prepare_user_assistant(output_path: str, model_name: str, max_length: int):
    # 1) load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 2) derive a char-limit heuristic  
    avg_chars_per_token = 4
    char_limit = avg_chars_per_token * max_length

    all_splits = {}

    for ds_id in DATASETS:
        print(f"\n🔍 Processing dataset: {ds_id}")
        ds_name = ds_id.split("/")[-1]
        builder = load_dataset(ds_id)
        splits  = list(builder.keys()) if isinstance(builder, dict) else ["train"]
        if ds_id != "nvidia/Llama-Nemotron-Post-Training-Dataset":
            splits = ["train"]

        for split in splits:
            print(f"→ {ds_name}:{split}", end=" … ")

            raw = load_dataset(ds_id, split=split)

            # ——— TEST MODE: limit to first 500 examples for faster dev ———
            if DEBUG:
                raw = raw.select(range(500))
                print(f"(test) processing only {len(raw)} examples", end=" … ")

            # ——— Stage 1: quick char-count filter ———
            def short_enough(ex):
                conv = to_conversation(ex, ds_id)
                total_len = len(conv["user"]) + len(conv["assistant"])
                return total_len <= char_limit

            stage1 = raw.filter(
                short_enough,
                batched=False,
            )
            print(f"after char-filter: {len(stage1)}/{len(raw)}", end=" … ")

            # ——— Stage 2: normalize to {user,assistant} ———
            conv = stage1.map(
                lambda ex: to_conversation(ex, ds_id),
                remove_columns=stage1.column_names,
            )
            print(f"mapped to user/assistant: {len(conv)} examples", end=" … ")

            # ——— Stage 3: accurate token-count filter ———
            def token_ok_batch(examples):
                texts = [
                    u + tokenizer.eos_token + a
                    for u, a in zip(examples["user"], examples["assistant"])
                ]
                toks = tokenizer(texts, add_special_tokens=False)
                return [len(ids) <= max_length for ids in toks["input_ids"]]

            stage2 = conv.filter(
                token_ok_batch,
                batched=True,
                batch_size=1000,
            )
            print(f"after token-filter: {len(stage2)}/{len(conv)}", end=" … ")

            # ——— Collect filtered conversation ———
            all_splits[f"{ds_name}_{split}"] = stage2
            print("done.")

    combined = DatasetDict(all_splits)
    combined.save_to_disk(output_path)
    print(f"\n✅ Saved to {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare & filter SFT data as user/assistant pairs")
    p.add_argument("--model_name", type=str, required=False, default="../llada_local_instruct",
                   help="HF model or tokenizer name/path for token-count filtering")
    p.add_argument("--max_length", type=int, default=8192,
                   help="Maximum allowed token length per example")
    p.add_argument("--out", type=str, default="./sft_data",
                   help="Directory to save the formatted & filtered DatasetDict")
    args = p.parse_args()

    print(f"Preparing user/assistant datasets into {args.out}, filtering > {args.max_length} tokens …")
    prepare_user_assistant(args.out, args.model_name, args.max_length)