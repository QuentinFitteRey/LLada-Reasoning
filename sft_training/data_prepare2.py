#!/usr/bin/env python3
import os
import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

# ——— Configuration defaults ———
# --- To run only for Orca, comment out the other datasets ---
DATASETS = [
#     "nvidia/Llama-Nemotron-Post-Training-Dataset",
#     "bespokelabs/Bespoke-Stratos-17k",
#     "open-thoughts/OpenThoughts-114k",
#     "databricks/databricks-dolly-15k",
    "Open-Orca/SlimOrca",
]
AVG_CHARS_PER_TOKEN = 4

def to_conversation(example, ds_id):
    if ds_id.startswith("nvidia/Llama-Nemotron"):
        user = " ".join(msg.get("content", "") for msg in example.get("input", []))
        return {"user": user, "assistant": example.get("output", "")}

    if ds_id == "databricks/databricks-dolly-15k":
        instr = example.get("instruction", "").strip()
        ctx = example.get("context", "").strip()
        prompt = f"{instr}\n\n{ctx}" if ctx else instr
        return {"user": prompt, "assistant": example.get("response", "")}

    # Corrected logic for Orca-style datasets
    if "conversations" in example:
        sys_p = ""
        users, assists = [], []
        for turn in example.get("conversations", []):
            who = turn.get("from", "").lower()
            txt = turn.get("value", "").strip()
            if who == "system":
                sys_p = txt
            elif who in ("user", "human"):
                users.append(txt)
            elif who in ("gpt", "assistant"):
                assists.append(txt)
        
        user_str = (sys_p + "\n\n") if sys_p else ""
        user_str += " ".join(users)
        return {"user": user_str, "assistant": " ".join(assists)}

    # fallback
    joined = "\n".join(f"{k}: {v}" for k, v in example.items())
    return {"user": joined, "assistant": ""}

def preprocess_batch(examples, ds_id, tokenizer, max_len, char_limit):
    # 1) build and char-filter conversations
    convs = []
    N = len(next(iter(examples.values())))
    for i in range(N):
        rec = {col: examples[col][i] for col in examples}
        c = to_conversation(rec, ds_id)
        if len(c["user"]) + len(c["assistant"]) <= char_limit:
            convs.append(c)

    # 2) batch-tokenize
    texts = [c["user"] + tokenizer.eos_token + c["assistant"] for c in convs]
    tok = tokenizer(texts, add_special_tokens=False)

    # 3) keep only those under max_len
    keeps = [i for i, ids in enumerate(tok["input_ids"]) if len(ids) <= max_len]

    return {
        "user":      [convs[i]["user"] for i in keeps],
        "assistant": [convs[i]["assistant"] for i in keeps],
    }

def process_dataset(ds_id, tokenizer, max_length, max_examples, num_proc, batch_size, debug):
    char_limit = AVG_CHARS_PER_TOKEN * max_length

    # 1) load raw
    try:
        raw = load_dataset(ds_id, "SFT")
    except:
        raw = load_dataset(ds_id)
    if not isinstance(raw, DatasetDict):
        raw = DatasetDict({"train": raw})

    # debug: cap each split to 500 examples
    if debug:
        for split in raw:
            n = min(500, len(raw[split]))
            raw[split] = raw[split].select(range(n))

    # 2) preslice: combine splits → train/test → cap 2×max_examples
    splits = dict(raw)
    val = splits.pop("validation", None) or splits.pop("valid", None) or splits.pop("test", None)
    train = concatenate_datasets(list(splits.values()))
    if val is None:
        tmp   = train.train_test_split(test_size=0.2, seed=42)
        train, val = tmp["train"], tmp["test"]
    capN = min(len(train), max_examples * 2)
    train = train.shuffle(seed=42).select(range(capN))

    # 3) filter by raw char count
    def char_ok(ex):
        c = to_conversation(ex, ds_id)
        return len(c["user"]) + len(c["assistant"]) <= char_limit

    train = train.filter(char_ok, num_proc=num_proc)
    val   = val.filter(char_ok,   num_proc=num_proc)

    # 4) map/tokenize & filter by token length
    train = train.map(
        lambda batch: preprocess_batch(batch, ds_id, tokenizer, max_length, char_limit),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=train.column_names,
    )
    val = val.map(
        lambda batch: preprocess_batch(batch, ds_id, tokenizer, max_length, char_limit),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=val.column_names,
    )

    # 5) cap final train to max_examples
    if not debug:
        train = train.shuffle(seed=42)
        if len(train) > max_examples:
            train = train.select(range(max_examples))
    # always shuffle validation for consistency
    val = val.shuffle(seed=42)

    return DatasetDict({"train": train, "validation": val})

def main():
    parser = argparse.ArgumentParser("SFT Data Prep (no caching)")
    parser.add_argument("--model_name",   type=str, default="./llada_local_instruct")
    parser.add_argument("--max_length",   type=int, default=8192)
    parser.add_argument("--max_examples", type=int, default=500000)
    parser.add_argument("--num_proc",     type=int, default=24)
    parser.add_argument("--batch_size",   type=int, default=1000)
    parser.add_argument("--out",          type=str, default="./sft_data_orca")
    parser.add_argument("--debug",        action="store_true",
                        help="cap each split to 500 examples for quick testing")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    combined = {}

    for ds_id in DATASETS:
        print(f"\n🔍 Processing dataset: {ds_id}")
        ds = process_dataset(
            ds_id,
            tokenizer,
            args.max_length,
            args.max_examples,
            args.num_proc,
            args.batch_size,
            args.debug,
        )
        name = ds_id.split("/")[-1]
        combined[f"{name}_train"]      = ds["train"]
        combined[f"{name}_validation"] = ds["validation"]

    # combine and save
    combined_dd  = DatasetDict(combined)
    out_dir      = os.path.join(args.out, "combined")
    os.makedirs(out_dir, exist_ok=True)
    combined_dd.save_to_disk(out_dir)
    print(f"\n✅ All datasets combined and saved to {out_dir}")

if __name__ == "__main__":
    main()