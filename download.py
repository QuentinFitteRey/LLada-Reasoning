import os
from huggingface_hub import snapshot_download

# Define the model ID and the local directory name
model_id = "GSAI-ML/LLaDA-8B-Base"
local_model_path = "./llada_local"

# Create the directory if it doesn't exist
os.makedirs(local_model_path, exist_ok=True)

# Download the model files to the local directory
print(f"Downloading model files to {local_model_path}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_model_path,
    local_dir_use_symlinks=False # Set to False to download files directly
)
print("Download complete.")