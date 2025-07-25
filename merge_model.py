from llada_local_15.modeling_llada import LLaDAModelLM
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from transformers import AutoTokenizer

def merge_lora_weights(model, lora_weights):
    """
    Merge LoRA weights into the base model.
    
    Args:
        model (LLaDAModelLM): The base model to merge LoRA weights into.
        lora_weights (str): Path to the LoRA weights to merge.
    
    Returns:
        LLaDAModelLM: The model with merged LoRA weights.
    """
    # Load the LoRA weights
    lora_model = PeftModel.from_pretrained(model, lora_weights)

    # Merge the LoRA weights into the base model
    merged_model = lora_model.merge_and_unload()
    
    return merged_model

def save_merged_model(model, tokenizer, output_dir):
    """
    Save the merged model and its tokenizer to the specified directory.
    
    Args:
        model (LLaDAModelLM): The model with merged LoRA weights.
        tokenizer (AutoTokenizer): The updated tokenizer.
        output_dir (str): Directory to save the merged model and tokenizer.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir) # <-- Add this crucial line
    print(f"Merged model and tokenizer saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge LoRA weights into LLaDAModelLM")
    parser.add_argument("--model_name", type=str, default="./llada_local_15", help="Path to the base model")
    parser.add_argument("--lora_weights", type=str, default="/home/hice1/yluo432/scratch/LLada-Reasoning/step-1100/sft_adapter", help="Path to the LoRA weights")
    parser.add_argument("--output_dir", type=str, default="./merged_model_final", help="Directory to save the merged model")

    args = parser.parse_args()
    
    # --- Step 1: Load the original base model and tokenizer ---
    print(f"Loading original base model from: {args.model_name}")
    model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # --- Step 2: Replicate the EXACT setup from the training script ---
    print("Replicating training setup: adding special tokens and resizing embeddings...")
    special_tokens_to_add = {
        "additional_special_tokens": ["<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>","<|begin_of_thought|>","<|end_of_thought|>" "<|begin_of_solution|>", "<|end_of_solution|>"]
    }
    # Note: There was a missing comma in your training script between </end_of_thought> and <|begin_of_solution|>.
    # Python treats this as string concatenation, so we do the same here to match perfectly.
    # Corrected special tokens list if needed:
    # "additional_special_tokens": ["<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|begin_of_thought|>","<|end_of_thought|>", "<|begin_of_solution|>", "<|end_of_solution|>"]
    
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|pad|>"
        
    tokenizer.add_special_tokens(special_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model and tokenizer resized. New vocabulary size: {len(tokenizer)}")
    model.to("cuda")

    # --- Step 3: Now merge the LoRA weights onto the correctly prepared model ---
    # The architectures now match, so this will succeed.
    merged_model = merge_lora_weights(model, args.lora_weights)
    
    # --- Step 4: Save the final, merged model ---
    save_merged_model(merged_model, tokenizer, args.output_dir)

if __name__ == "__main__":
    main()