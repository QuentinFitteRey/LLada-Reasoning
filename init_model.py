
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict, get_peft_model
import os

adapter_default = os.path.expanduser("~/scratch/LLaDA_checkpoints/test_checkpoint")

def init_model(
    model_path: str = "./llada_local",
    adapter_path: str | None = None,
    load_lora: bool = False,
    device: str = "cuda",
    torch_dtype=None,
    trust_remote_code: bool = True,
    local_files_only: bool = True,
    attn_implementation: str = None,
    device_map: dict | str | None = "auto",
):
    # Path to the local directory containing the modified model
    local_model_path = model_path

    print(f"Loading tokenizer from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only
    )

    if load_lora:
        # Instruct + Reasoning special tokens
        special_tokens_to_add = {
            "additional_special_tokens": ["<|mdm_mask|>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>","<|begin_of_thought|>","<|end_of_thought|>" "<|begin_of_solution|>", "<|end_of_solution|>"]
        }
    else:
        # Instruct special tokens only
        special_tokens_to_add = {
            "additional_special_tokens": []
        }

    if tokenizer.pad_token is None:
        print("No pad token found, adding <|pad|> as pad token.")
        special_tokens_to_add["pad_token"] = "<|pad|>"

    print(f"len(tokenizer): \n {len(tokenizer)}")

    # Add tokens to tokenizer
    tokenizer.add_special_tokens(special_tokens_to_add)

    print(f"Loading modified model from: {local_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )

    # Resize embeddings of the entire PeftModel
    model.resize_token_embeddings(len(tokenizer))

    print(f"len(tokenizer): \n {len(tokenizer)}")

    print("Model loaded successfully with local modifications.")

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    if load_lora or adapter_path:
        real_adapter = adapter_path or adapter_default
        print(f"🔗 Loading LoRA adapter from {real_adapter} …")
        # this will raise if real_adapter isn’t actually a PEFT folder

        # validate that adapter really is a peft folder
        try:
            _ = PeftConfig.from_pretrained(real_adapter)
            model = PeftModel.from_pretrained(model, real_adapter)
            print("LoRA adapter loaded.")
        except Exception as e:
            print("Could not load adapter:", e)
    else: # Add an empty adapter if not loading LoRA
        print("No LoRA adapter specified, using empty adapter.")
        config = LoraConfig(
            r=8, # Rank can be small, it doesn't matter much since it will be zeroed out
            lora_alpha=16,         target_modules=[            "q_proj",             "k_proj",             "v_proj",             "o_proj",             "gate_proj",             "up_proj", "down_proj", ], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", )
        model = get_peft_model(model, config)

    return model, tokenizer

if __name__ == "__main__":
    init_model()