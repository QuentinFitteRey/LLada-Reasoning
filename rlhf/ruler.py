import asyncio
from art import Trajectory, TrajectoryGroup
from art.rewards import ruler_score_group
from new_generation import generate_with_dual_cache
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from transformers import AutoTokenizer
import torch
from llada_local.modeling_llada import LLaDAModelLM  # Assuming this is the correct import for
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # required dummy

def init_model():
    """
    Initializes and returns the base model and tokenizer.
    """
    local_model_path = "./llada_local_1.5"

    print(f"Loading tokenizer from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    print(f"Loading base model from: {local_model_path}")
    print("Base model loaded successfully.")
    model = LLaDAModelLM.from_pretrained(
        local_model_path,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )
    special_tokens_to_add = {
        "additional_special_tokens": ["<|mdm_mask|>", "<think>", "</think>", "<|start_header_id|>", "<|end_header_id|>","<|eot_id|>"]
    }
    if tokenizer.pad_token is None:
        special_tokens_to_add["pad_token"] = "<|pad|>"
    tokenizer.add_special_tokens(special_tokens_to_add)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def generate_response(model, tokenizer, prompt, device='cuda', steps=128, gen_length=128, block_length=32,
                      temperature=0.0, remasking='low_confidence'):
    input_ids = torch.tensor(tokenizer(prompt)['input_ids']).to(device).unsqueeze(0)
    out, _ = generate_with_dual_cache(
        model, input_ids,
        steps=steps, gen_length=gen_length, block_length=block_length,
        temperature=temperature, remasking=remasking
    )
    text = tokenizer.batch_decode(
        out[:, input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    return text

async def make_group(data, model, tokenizer, num_extra=3, judge_model="ollama/qwen3:8b", **ruler_kwargs):
    prompt = data["prompt"]
    original_answer = data["answer"]
    
    # Setup the shared system/user prompt
    messages = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user", "content":prompt},
    ]

    trajectories = []

    # Add the original answer as the first trajectory
    
    trajectories.append(
        Trajectory(messages_and_choices=[
            *messages,
            Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(
                role="assistant",
                content=original_answer
            ))
        ], reward=0.0)
    )

    # Generate and add 3 new completions
    for _ in range(num_extra):
        out = generate_response(model, tokenizer, prompt, temperature=0.7)
        trajectories.append(
            Trajectory(messages_and_choices=[
                *messages,
                Choice(finish_reason="stop", index=0, message=ChatCompletionMessage(
                role="assistant",
                content=out
            ))
            ], reward=0.0)
        )

    group = TrajectoryGroup(trajectories)
    judged = await ruler_score_group(group, judge_model, **ruler_kwargs)
    if judged:
        sorted_trajectories = sorted(judged.trajectories, key=lambda t: t.reward, reverse=True)
        for rank, traj in enumerate(sorted_trajectories, 1):
            messages = traj.messages()
            print(f"Rank {rank}: Score {traj.reward:.3f}")
            print(f"  Response: {messages[-1]['content']}...")
    return judged

if __name__ == "__main__":
    data = {
        "prompt": "what is German Idealism?",
        "answer": "German Idealism is a philosophical movement that emerged in Germany in the late 18th and early 19th centuries, following the work of Immanuel Kant. It centers on the idea that reality is fundamentally shaped by the mind—that is, the structures of thought play a key role in how we experience the world."
    }
    model, tokenizer = init_model()
    model = model.to("cuda")
    model.eval()
    asyncio.run(make_group(data, model, tokenizer))