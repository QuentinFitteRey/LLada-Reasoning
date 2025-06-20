from transformers import AutoModel, AutoTokenizer
import torch

# Path to your local directory containing the modified model
local_model_path = "./llada_local" 

print(f"Loading tokenizer from: {local_model_path}")
model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
print("Model loaded successfully with local modifications.")
