from transformers import AutoModel, AutoTokenizer
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
        # Add other parameters as needed, e.g., for quantization
        # device_map="auto",
        # load_in_8bit=True 
    )
    print("Model loaded successfully with local modifications.")
    return model, tokenizer

if __name__ == "__main__":
    init_model()
