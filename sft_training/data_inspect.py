#!/usr/bin/env python3
import argparse
import itertools
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

DATASETS = [
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "bespokelabs/Bespoke-Stratos-17k",
    "open-thoughts/OpenThoughts-114k",
    "databricks/databricks-dolly-15k",
    "Open-Orca/SlimOrca",
]

def inspect_per_split():
    for ds_id in DATASETS:
        # 1) Load all splits in streaming mode
        ds_dict = load_dataset(ds_id, streaming=True)
        splits = list(ds_dict.keys())
        print(f"\n=== {ds_id} → splits found: {splits} ===")
        # 2) For each split, show 3 examples
        for split in splits:
            print(f"\n--- inspecting split '{split}' (3 examples) ---")
            stream = ds_dict[split]
            for i, ex in enumerate(itertools.islice(stream, 3), start=1):
                print(f"\nExample {i} keys: {list(ex.keys())}")
                for k, v in ex.items():
                    snippet = repr(v)
                    print(f"  {k!r}: {snippet[:200]}")
                print("-"*30)

if __name__ == "__main__":
    inspect_per_split()
    print("\nDone inspecting")