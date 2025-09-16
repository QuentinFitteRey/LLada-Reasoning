import os
from huggingface_hub import snapshot_download

model_id = "GSAI-ML/LLaDA-8B-Base"
local_model_path = "./llada_local"

os.makedirs(local_model_path, exist_ok=True)

print(f"Downloading model files to {local_model_path}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_model_path,
    local_dir_use_symlinks=False 
)
print("Download complete.")