import os
from huggingface_hub import snapshot_download

# Define the model ID and your local directory name
model_id = "GSAI-ML/LLaDA-8B-Instruct"
local_model_path = "./llada_local_instruct"

# Create the directory if it doesn't exist
os.makedirs(local_model_path, exist_ok=True)

# Download the model files to your local directory
print(f"Downloading model files to {local_model_path}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_model_path,
    local_dir_use_symlinks=False # Set to False to download files directly
)
print("Download complete.")