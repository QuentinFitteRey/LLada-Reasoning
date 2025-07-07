from llada_local.modeling_llada import LLaDAModelLM
from peft import LoraConfig, get_peft_model, PeftModel
import torch


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

def save_merged_model(model, output_dir):
    """
    Save the merged model to the specified directory.
    
    Args:
        model (LLaDAModelLM): The model with merged LoRA weights.
        output_dir (str): Directory to save the merged model.
    """
    model.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge LoRA weights into LLaDAModelLM")
    parser.add_argument("--model_name", type=str, default="./llada_local", help="Path to the base model")
    parser.add_argument("--lora_weights", type=str, default="/home/hice1/qfitterey3/scratch/LLada-Reasoning/checkpoints/checkpoints_llada_pretrain_8k_clip_lr15_betterval_128_good_newdata/step-320", help="Path to the LoRA weights")
    parser.add_argument("--output_dir", type=str, default="./merged_model", help="Directory to save the merged model")

    args = parser.parse_args()
    
    # Load the base model
    model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.to("cuda")
    
    # Merge LoRA weights
    merged_model = merge_lora_weights(model, args.lora_weights)
    
    # Save the merged model
    save_merged_model(merged_model, args.output_dir)
    
if __name__ == "__main__":
    main()