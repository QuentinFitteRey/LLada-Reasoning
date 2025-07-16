from datasets import load_dataset

# Step 1: Define the mapping function
def map_shp_to_chosen_rejected(example):
    prompt = example["history"]
    if example["labels"] == 1:
        return {
            "prompt": prompt,
            "chosen": example["human_ref_B"],
            "rejected": example["human_ref_A"]
        }
    else:
        return {
            "prompt": prompt,
            "chosen": example["human_ref_A"],
            "rejected": example["human_ref_B"]
        }

# Step 2: Load a subset of SHP (first 10,000 examples from the train split)
print("Loading dataset...")
shp_subset = load_dataset("stanfordnlp/SHP", split="train[:100]")

# Step 3: Apply the mapping
print("Applying mapping...")
shp_processed = shp_subset.map(map_shp_to_chosen_rejected)

# Step 4: Save the processed dataset locally
save_path = "./SHP_train_subset_processed"
print(f"Saving processed dataset to {save_path} ...")
shp_processed.save_to_disk(save_path)

print("✅ Done!")
