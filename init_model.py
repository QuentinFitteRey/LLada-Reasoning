
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def init_model():
    # Path to your local directory containing the modified model
    local_model_path = "./llada_local" 


    print(f"Loading tokenizer from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    print(f"Loading modified model from: {local_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,  # Automatically set the dtype based on the model
       #device_map="auto",  # Automatically set the device map
        # Add other parameters as needed, e.g., for quantization
        # device_map="auto",
        # load_in_8bit=True 
    )
    print("Model loaded successfully with local modifications.")
    return model, tokenizer

if __name__ == "__main__":
    init_model()
