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
    lora_model = PeftModel.from_pretrained(model, lora_weights)

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
    tokenizer.save_pretrained(output_dir) 
    print(f"Merged model and tokenizer saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge LoRA weights into LLaDAModelLM")
    parser.add_argument("--model_name", type=str, default="./llada_local_15", help="Path to the base model")
    parser.add_argument("--lora_weights", type=str, default="/home/hice1/yluo432/scratch/LLada-Reasoning/step-1100/sft_adapter", help="Path to the LoRA weights")
    parser.add_argument("--output_dir", type=str, default="./merged_model_final", help="Directory to save the merged model")

    args = parser.parse_args()
    
    print(f"Loading original base model from: {args.model_name}")
    model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    print("Replicating training setup: adding special tokens and resizing embeddings...")
    special_tokens_to_add = {
        "additional_special_tokens": ["<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>","<|begin_of_thought|>","<|end_of_thought|>" "<|begin_of_solution|>", "<|end_of_solution|>"]
    }
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|pad|>"
        
    tokenizer.add_special_tokens(special_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model and tokenizer resized. New vocabulary size: {len(tokenizer)}")
    model.to("cuda")

    merged_model = merge_lora_weights(model, args.lora_weights)
    
    save_merged_model(merged_model, tokenizer, args.output_dir)

if __name__ == "__main__":
    main()