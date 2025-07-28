
import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from transformers import AutoTokenizer
from init_model import init_model # Assuming this is a local helper script


def add_gumbel_noise(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature == 0:
        return logits
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return (logits / temperature) + gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens
@torch.no_grad()
def generate_with_dual_cache(model, prompt_ids, steps=128, gen_length=128, block_length=128, temperature=0.,
                             remasking='low_confidence', mask_id=126336, threshold=None):
    """
    Dual-cache generation with batch support.
    """
    batch_size, prompt_len = prompt_ids.shape
    device = model.device

    total_len = prompt_len + gen_length
    x = torch.full((batch_size, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.clone()

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by number of blocks"
    sub_steps = steps // num_blocks

    nfe = 0
    # 1) Prime cache on full sequence once
    output = model(x, use_cache=True)
    past_key_values = output.past_key_values

    for block_idx in range(num_blocks):
        # Define global positions for this block
        start = prompt_len + block_idx * block_length
        end = start + block_length

        # Pre-compute transfer counts for this block
        mask_block = (x[:, start:end] == mask_id)
        num_trans = get_num_transfer_tokens(mask_block, sub_steps)

        # Initial fill for this block (first sub-step)
        mask_full = (x == mask_id)
        mask_full[:, end:] = False
        
        # --- MODIFICATION HERE ---
        # Always pass the number of tokens to transfer. The threshold will act as a filter.
        x0, transfer_idx = get_transfer_index(
            output.logits, temperature, remasking,
            mask_full, x,
            num_trans[:, 0], # Always pass num_trans
            threshold
        )
        x[transfer_idx] = x0[transfer_idx]
        nfe += 1

        # Iterative refinement within the block
        for step in range(1, sub_steps):
            nfe += 1
            block_x = x[:, start:end]
            block_mask = (block_x == mask_id)

            if not block_mask.any():
                break

            replace_mask = torch.zeros((batch_size, total_len), dtype=torch.bool, device=device)
            replace_mask[:, start:end] = True

            out = model(
                block_x,
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_mask
            )
            logits = out.logits
            past_key_values = out.past_key_values

            # --- MODIFICATION HERE ---
            # Always pass the number of tokens to transfer.
            x0_blk, transfer_blk = get_transfer_index(
                logits, temperature, remasking,
                block_mask, block_x,
                num_trans[:, step], # Always pass num_trans
                threshold
            )
            
            x_block_view = x[:, start:end]
            x_block_view[transfer_blk] = x0_blk[transfer_blk]

    return x, nfe

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    """
    Selects tokens to unmask based on confidence, respecting the number of tokens per step
    and applying an optional confidence threshold.
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(remasking)

    # Only consider tokens that were originally masked for confidence scoring
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -torch.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)

    # This loop is safe for batching
    for j in range(confidence.shape[0]):
        # Determine how many tokens to consider unmasking in this step
        num_to_consider = num_transfer_tokens[j].item()

        # Ensure k for topk is not larger than the number of remaining masked elements
        k = min(num_to_consider, mask_index[j].sum().item())
        if k == 0:
            continue
            
        # Get the top k most confident predictions among the masked tokens
        vals, sel = torch.topk(confidence[j], k=k)
        
        # If a threshold is provided, filter the selected indices
        if threshold is not None:
            # Only keep tokens where the confidence (vals) is >= threshold
            above_threshold_mask = (vals >= threshold)
            sel = sel[above_threshold_mask]
        
        # If any tokens remain after filtering, mark them for transfer
        if sel.numel() > 0:
            transfer_index[j, sel] = True
            
    return x0, transfer_index


def main():
    device = 'cuda'
    model, tokenizer = init_model(lora=True)
    model = model.to(device)
    think = False
    
    base_prompt_instruction = "You must perform a detailed, step-by-step thinking process to solve the problem. Your thinking should be a comprehensive cycle of analysis, exploration, and self-correction. Engage in reflection, back-tracing to refine errors, and iteration to develop a well-considered path to the solution. Put this entire process between <think> and </think> tags. \nAfter the closing </think> tag, present your final answer. Your answer should begin with the conclusion, followed by a brief summary that explains how you arrived at it by referencing the key steps from your thinking process.\n" if think else "You are not required to have detailed thinking on the problem between <think> and </think> tags. \nYou can provide a direct answer to the question without detailed thinking. \nYou can still take steps to solve the problem, but you do not need to provide detailed thinking on the problem.\n"

    # Define multiple prompts for batching
    prompts_content = [
        """
        You are given the following classic logic puzzle, often called the Zebra Puzzle or Einstein’s Riddle.
        Your task is to determine Who owns the zebra? and Who drinks water? based on the following clues:
        1. The Brit lives in the red house.
        2. The Swede keeps dogs as pets.
        3. The Dane drinks tea.
        4. The green house is immediately to the left of the white house.
        5. The green house’s owner drinks coffee.
        6. The person who smokes Pall Mall keeps birds.
        7. The owner of the yellow house smokes Dunhill.
        8. The man living in the center house drinks milk.
        9. The Norwegian lives in the first house.
        10. The man who smokes Blend lives next to the one who keeps cats.
        11. The man who keeps horses lives next to the man who smokes Dunhill.
        12. The man who smokes BlueMaster drinks beer.
        13. The German smokes Prince.
        14. The Norwegian lives next to the blue house.
        15. The man who smokes Blend has a neighbor who drinks water.
        """,
        "You have a 3-gallon jug and a 5-gallon jug, neither of which has any measurement markings, and an unlimited water supply. Using only these two jugs, measure out exactly 4 gallons of water. This should take 6 steps. If you have more than 6 steps, you can restart your logic. A gallon jug can not have more than it's capacity.\n",
        "A number consists of two digits. The digit in the tens place is three times the digit in the units place. If you reverse the digits, the new number is 36 less than the original number. What is the original number? You have to ouput the answer at the end even if you are not sure about the answer. \n"
    ]
    
    full_prompts = [base_prompt_instruction + p for p in prompts_content]
    print(f"Processing a batch of {len(full_prompts)} prompts.")

    chat_formatted_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
        for p in full_prompts
    ]
    
    inputs = tokenizer(
        chat_formatted_prompts,
        return_tensors='pt',
        padding=True,
    ).to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    mask_token_id = 126336

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    
    out, _ = generate_with_dual_cache(
        model, input_ids,
        steps=2048, gen_length=2048, block_length=128,
        temperature=0.0, remasking='low_confidence', threshold=0.2,
        mask_id=mask_token_id
    )
    
    t1.record()
    torch.cuda.synchronize()
    print(f"Generation time for batch: {t0.elapsed_time(t1)/1000:.2f}s")
    
    prompt_lengths = attention_mask.sum(dim=1)
    
    decoded_texts = []
    for i in range(out.shape[0]):
        generated_ids = out[i,]
        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        decoded_texts.append(text)

    for i, text in enumerate(decoded_texts):
        print(f"\n--- Result for Prompt {i+1} ---")
        print(text)


if __name__ == '__main__':
    main()