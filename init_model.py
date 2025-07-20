
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict

def init_model(lora=False):
    # Path to your local directory containing the modified model
    local_model_path = "/home/hice1/qfitterey3/scratch/LLada-Reasoning/llada_local_1.5" 
    local_model_tokenizer_path = "/home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora/step-600"

    print(f"Loading tokenizer from: {local_model_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_tokenizer_path, use_fast=True, local_files_only=True)

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
    )
    print("Model loaded successfully with local modifications.")
    print(model)
    # special_tokens_to_add = {
    # "additional_special_tokens": ["<|mdm_mask|>", "<think>", "</think>"]
    # }

    # if tokenizer.pad_token is None:
    #     special_tokens_to_add["pad_token"] = "<|pad|>"

    # # Add tokens to tokenizer
    # tokenizer.add_special_tokens(special_tokens_to_add)

    # # Resize embeddings of the entire PeftModel
    model.resize_token_embeddings(len(tokenizer))
    print(len(tokenizer))

    if lora:
        lora_config = PeftConfig.from_pretrained(
            "/home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora/step-600/sft_adapter"
        )
        print("Loading LoRA configuration...")
        model = PeftModel.from_pretrained(model, "/home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_nemotron_15_4_goodlora/step-600/sft_adapter", lora_config=lora_config)
        print("LoRA model loaded successfully.")
    return model, tokenizer

if __name__ == "__main__":
    init_model()
