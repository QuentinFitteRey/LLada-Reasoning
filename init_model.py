
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
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
    # Path to your local directory containing the modified model
    local_model_path = model_path

    print(f"Loading tokenizer from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only
    )

    print(f"Loading modified model from: {local_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
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
        from peft import PeftConfig, PeftModel

        # validate that adapter really is a peft folder
        try:
            _ = PeftConfig.from_pretrained(real_adapter)
            model = PeftModel.from_pretrained(model, real_adapter)
            print("LoRA adapter loaded.")
        except Exception as e:
            print("Could not load adapter:", e)

    return model, tokenizer

if __name__ == "__main__":
    init_model()
