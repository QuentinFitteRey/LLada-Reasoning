import torch
import numpy as np
import torch.nn.functional as F
from typing import Literal
from transformers import AutoTokenizer, AutoModel
from init_model import init_model


# def add_gumbel_noise(logits, temperature):
#     '''
#     The Gumbel max is a method for sampling categorical distributions.
#     According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
#     Thus, we use float64.
#     '''
#     if temperature == 0:
#         return logits
#     logits = logits.to(torch.float64)
#     noise = torch.rand_like(logits, dtype=torch.float64)
#     gumbel_noise = (- torch.log(noise)) ** temperature
#     return logits.exp() / gumbel_noise

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Adds Gumbel noise to logits for stochastic sampling.
    The Gumbel-Max trick states that argmax(logits + Gumbel(0,1)) is a sample
    from the categorical distribution given by softmax(logits).
    Scaling the logits by temperature controls the 'peakiness' of the distribution.
    """
    if temperature == 0:
        return logits
    # Note: No need for float64 if we avoid .exp(). float32 is fine.
    # The Gumbel noise is -log(-log(U)) where U ~ Uniform(0,1)
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-9) + 1e-9) # Add epsilon for stability
    return (logits / temperature) + gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


# @ torch.no_grad()
# def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336):
#     '''
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The toke id of [MASK] is 126336.
#     '''
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :prompt.shape[1]] = prompt.clone()

#     prompt_index = (x != mask_id)

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length

#     assert steps % num_blocks == 0
#     steps = steps // num_blocks

#     for num_block in range(num_blocks):
#         block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#         for i in range(steps):
#             mask_index = (x == mask_id)
#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 logits = model(x_).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x).logits

#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

#             if remasking == 'low_confidence':
#                 p = F.softmax(logits, dim=-1)
#                 x0_p = torch.squeeze(
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
#             elif remasking == 'random':
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)

#             x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)

#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):
#                 _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]

#     return x

def generate(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.7,
    cfg_scale: float = 4.0,
    remasking: Literal['low_confidence', 'random'] = 'low_confidence',
    mask_id: int = 126336
) -> torch.Tensor:
    """
    Generates text using a semi-autoregressive iterative decoding process.

    Args:
        model: The mask predictor model.
        prompt: A tensor of shape (1, L_prompt).
        steps: Total sampling steps for the entire generation process.
        gen_length: The total number of tokens to generate.
        block_length: The size of each generation block for semi-autoregressive decoding.
        temperature: Sampling temperature for Gumbel-Max. 0 means deterministic.
        cfg_scale: Classifier-Free Guidance scale. 0 disables CFG.
        remasking_strategy: Strategy to choose which tokens to keep.
        mask_id: The token id for the [MASK] token.
    """
    device = model.device
    batch_size, prompt_len = prompt.shape
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks
    x = torch.full((batch_size, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone()
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, :prompt_len] = True
    for i in range(num_blocks):
        block_start = prompt_len + i * block_length
        block_end = block_start + block_length
        
        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, block_start:block_end] = True
        
        num_tokens_to_generate_in_block = block_mask.sum()
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            masked_positions = (x == mask_id)

            if cfg_scale > 0.:
                cond_logits = model(x).logits

                uncond_x = x.clone()
                uncond_x[prompt_index] = mask_id
                uncond_logits = model(uncond_x).logits
                
                logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            else:
                logits = model(x).logits

            noisy_logits = add_gumbel_noise(logits, temperature)
            predicted_tokens = torch.argmax(noisy_logits, dim=-1)
            
            predicted_tokens = torch.where(masked_positions, predicted_tokens, x)

            if remasking == 'low_confidence':
                token_probs = F.softmax(logits, dim=-1)
                confidence = torch.gather(token_probs, -1, predicted_tokens.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                confidence = torch.rand_like(predicted_tokens, dtype=torch.float32)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking_strategy}' not implemented.")


            confidence = torch.where(masked_positions, confidence, -1.0)
            confidence[prompt_index] = -1.0 
            

            k = num_transfer_tokens[:, step].item() # .item() assumes batch_size=1
            if k == 0:
                continue

            _, topk_indices = torch.topk(confidence.view(-1), k=k)
            
            x_flat = x.view(-1)
            predicted_tokens_flat = predicted_tokens.view(-1)
            
            x_flat[topk_indices] = predicted_tokens_flat[topk_indices]
            x = x_flat.view(x.shape)

    return x

def main():
    device = 'cuda'

    model, tokenizer = init_model(lora=False)
    model = model.to(device)

    # Plain text prompt for base model
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. To do 8 km she is running "
    #prompt = "Albert Einstein was born in Ulm, Germany, on"
    # Tokenize the prompt directly
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Generate output
    out = generate(
        model,
        input_ids,
        steps=128,
        gen_length=128,
        block_length=16,
        temperature=0.0,
        cfg_scale=0.0,
        remasking='low_confidence'
    )

    # Decode and print only the generated part
    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(generated_text)



if __name__ == '__main__':
    main()
