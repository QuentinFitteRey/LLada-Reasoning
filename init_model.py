
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict, get_peft_model
import os

adapter_path = os.path.expanduser("~/scratch/LLaDA_checkpoints/test_checkpoint")

def init_model(lora=False):
    # Path to your local directory containing the modified model
    local_model_path = "./llada_local_1.5"  # Adjust this path as needed


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
        local_files_only=True,  # Ensure it loads from local files only
        # attn_implementation = "flash_attention_2"
    )
    print("Model loaded successfully with local modifications.")
    lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  
        )
    model = get_peft_model(model, lora_config)
    if lora:
        print("Loading LoRA configuration...")
        model = PeftModel.from_pretrained(model, adapter_path, lora_config=lora_config)
        print("LoRA model loaded successfully.")
    return model, tokenizer

if __name__ == "__main__":
    init_model()
