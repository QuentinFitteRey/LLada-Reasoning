import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import os
from new_generation import generate_with_dual_cache, get_transfer_index

NOT_THINKING_MODE = "You are not required to have detailed thinking on the problem between <think> and </think> tags. \nYou can provide a direct answer to the question without detailed thinking. \nYou can still take steps to solve the problem, but you do not need to provide detailed thinking on the problem.\n"

def map_shp_preformatted_prompt(example, tokenizer, max_len=4096):
    """
    Pre-formats the prompt according to the chat template and filters out long examples.
    """

    user_prompt = example["history"]

    formatted_prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt.strip()}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    if example["labels"] == 1:
        chosen_response = example["human_ref_B"]
        rejected_response = example["human_ref_A"]
    else:
        chosen_response = example["human_ref_A"]
        rejected_response = example["human_ref_B"]

    chosen_full = formatted_prompt + chosen_response.strip() + "<|eot_id|>"
    rejected_full = formatted_prompt + rejected_response.strip() + "<|eot_id|>"

    # Tokenize to check length
    chosen_len = len(tokenizer(chosen_full, add_special_tokens=False)["input_ids"])
    rejected_len = len(tokenizer(rejected_full, add_special_tokens=False)["input_ids"])

    # Filter if either is too long
    if chosen_len > max_len or rejected_len > max_len:
        return None

    return {
        "prompt": formatted_prompt,
        "chosen": chosen_response.strip() + "<|eot_id|>",
        "rejected": rejected_response.strip() + "<|eot_id|>"
    }

import re

def map_hh_rlhf_dpo(example, tokenizer, max_len=4096):
    chosen_full_text = example["chosen"]
    rejected_full_text = example["rejected"]

    # Extract shared prompt and responses
    split_chosen = chosen_full_text.rsplit("\n\nAssistant:", 1)
    split_rejected = rejected_full_text.rsplit("\n\nAssistant:", 1)

    if len(split_chosen) != 2 or len(split_rejected) != 2:
        return None  # skip malformed examples

    prompt_text = split_chosen[0]
    chosen_response = split_chosen[1].strip()
    rejected_response = split_rejected[1].strip()

    # Match sequences of form "\n\nHuman: ..." or "\n\nAssistant: ..."
    # and extract all roles and their messages in order
    pattern = r"(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|\Z)"
    matches = re.findall(pattern, prompt_text, re.DOTALL)

    # Convert to chat template format
    messages = []
    for role, content in matches:
        role_tag = "user" if role == "Human" else "assistant"
        messages.append({"role": role_tag, "content": content.strip()})

    # Format the prompt using tokenizer's chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Append <eos> to each response
    chosen_final = chosen_response + tokenizer.eos_token
    rejected_final = rejected_response + tokenizer.eos_token

    # Length filtering
    chosen_len = len(tokenizer(formatted_prompt + chosen_final, add_special_tokens=False)["input_ids"])
    rejected_len = len(tokenizer(formatted_prompt + rejected_final, add_special_tokens=False)["input_ids"])
    if chosen_len > max_len or rejected_len > max_len:
        return None

    return {
        "prompt": formatted_prompt,
        "chosen": chosen_final,
        "rejected": rejected_final,
    }

def map_hh_golden_rlhf(example, tokenizer, max_len=4096):
    """
    Maps the example to the format expected by the DPO trainer.
    """
    chosen_full_text = example["chosen"]
    rejected_full_text = example["rejected"]
    prompt_text = NOT_THINKING_MODE + example["prompt"]

    chosen_len = len(tokenizer(prompt_text + chosen_full_text, add_special_tokens=False)["input_ids"])
    rejected_len = len(tokenizer(prompt_text + rejected_full_text, add_special_tokens=False)["input_ids"])
    if chosen_len > max_len or rejected_len > max_len:
        return None
    return {
        "prompt": prompt_text,
        "chosen": chosen_full_text,
        "rejected": rejected_full_text,
    }

def init_model():
    local_model_path = "./llada_local_1.5"
    ADAPTER_PATH = os.path.expanduser("~/scratch/LLaDA_checkpoints/test_checkpoint")
    # Replace this with your model/tokenizer loading logic
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base_model = AutoModelForCausalLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    # 3. Apply LoRA adapters to create the PEFT model
    peft_model = get_peft_model(base_model, lora_config)
    peft_model = PeftModel.from_pretrained(peft_model, ADAPTER_PATH, lora_config=lora_config)
    return peft_model, tokenizer

def generate_response(model, tokenizer, prompt, device):
    input_ids = torch.tensor(tokenizer(prompt)['input_ids']).to(device).unsqueeze(0)

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()

    out, _ = generate_with_dual_cache(
        model, input_ids,
        steps=128, gen_length=128, block_length=32,
        temperature=0.0, remasking='low_confidence'
    )

    t1.record(); torch.cuda.synchronize()
    print(f"Generation time: {t0.elapsed_time(t1)/1000:.2f}s")

    generated = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    return generated

def process_dataset(dataset, model, tokenizer, device, num_samples):
    processed = []
    for example in tqdm(dataset.select(range(min(num_samples, len(dataset))))):
        prompt = example["prompt"]
        chosen = example["response"]

        try:
            rejected = generate_response(model, tokenizer, prompt, device)
        except Exception as e:
            print(f"Skipping due to error: {e}")
            continue

        processed.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    return processed

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to your dataset (HF name or local)")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save output JSONL")
    parser.add_argument("--num_samples", type=int, default=100, help="How many prompts to process")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = init_model()
    model = model.to(device)
    model.eval()
    model.gradient_checkpointing_enable()

    # Load dataset (assumes 'prompt' and 'response' keys)
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    
    # Process
    processed_data = process_dataset(dataset, model, tokenizer, device, args.num_samples)

    # Save as JSONL
    with open(args.output_path, "w", encoding="utf-8") as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
