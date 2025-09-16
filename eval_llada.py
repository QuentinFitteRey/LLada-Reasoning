import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModel
from generation import generate_with_dual_cache
from init_model import init_model
from accelerate import Accelerator

import io
import sys
import json
import datetime
import os
from contextlib import redirect_stdout
from functools import partial


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
    self,
    model_path: str = "./llada_local",
    adapter_path: str | None = None,
    load_lora: bool = False,
        mask_id=126336,
        max_length=8192,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=128,
        remasking='low_confidence',
        device="cuda",
        generate_batch_size=1,
        repetition_penalty=1.2,
        use_thinking=True,
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.load_lora = load_lora
        self.task = os.environ.get("EVAL_TASKS", None)
        self.generate_batch_size = generate_batch_size
        self.repetition_penalty = repetition_penalty
        self.use_thinking = use_thinking

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        print(f"[LLaDAEvalHarness] Loading model from: {self.model_path}"
              f"{' + adapter=' + self.adapter_path if self.adapter_path else ''}"
              f" (LoRA={'yes' if self.load_lora else 'no'})")
        model, tokenizer = init_model(
            model_path=self.model_path,
            adapter_path=self.adapter_path,
            load_lora=self.load_lora,
            torch_dtype=torch.bfloat16,
        )
        self.model = model
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.model.eval()
        print("number of steps:", steps)
        print("gen_length:", gen_length)
        print("block_length:", block_length)
        print("remasking:", remasking)


        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.device = torch.device(device)
            self.model = self.model.to(self.device)

        self.mask_id = mask_id

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking 	
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    def get_logits(self, batch, prompt_index):
        with torch.inference_mode():
            if self.cfg > 0.:
                assert len(prompt_index) == batch.shape[1]
                prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
                un_batch = batch.clone()
                un_batch[prompt_index] = self.mask_id
                batch = torch.cat([batch, un_batch])

            logits = self.model(batch).logits

            if self.cfg > 0.:
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (self.cfg + 1) * (logits - un_logits)
            return logits[:, :batch.shape[1]]

    def get_loglikelihood(self, prefix, target):
        with torch.inference_mode():
            seq = torch.concatenate([prefix, target])[None, :]
            seq = seq.repeat((self.batch_size, 1)).to(self.device)

            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

            loss_acc = []
            for _ in range(self.mc_num // self.batch_size):
                perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

                mask_indices = perturbed_seq == self.mask_id

                logits = self.get_logits(perturbed_seq, prompt_index)

                loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
                loss = loss.sum() / self.batch_size
                loss_acc.append(loss.item())

            return - sum(loss_acc) / len(loss_acc)

    def suffix_greedy_prediction(self, prefix, target):
        with torch.inference_mode():
            if not self.is_check_greedy:
                return False

            seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
            prefix, target = prefix.to(self.device), target.to(self.device)
            seq[0, :len(prefix)] = prefix

            for i in range(len(target)):
                mask_index = (seq == self.mask_id)
                logits = self.get_logits(seq, prompt_index)[mask_index]
                x0 = torch.argmax(logits, dim=-1)

                p = torch.softmax(logits.to(torch.float32), dim=-1)
                confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
                _, index = torch.sort(confidence, descending=True)
                x0[index[1:]] = self.mask_id
                seq[mask_index] = x0.clone()
            correct = target == seq[0, len(prefix):]
            correct = torch.all(correct)
            return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        if max(prompt_len) > 4096:
            print(f"[Warning] Some sequences are longer than 4096 tokens, truncating to 4096.")
            ds = ds.map(
                lambda x: {
                    "prefix": x["prefix"][:4096],
                    "target": x["target"][:4096 - len(x["prefix"])],
                },
                batched=False,
            )
            prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.inference_mode():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def llada_generate_until_tokenize(self, e, idx, tokenizer):
        return {
            "question": tokenizer(e["question"])["input_ids"],
            "question_text": e["question"],
            "until": e["until"],
        }

    def generate_until_not_batched(self, requests: list[Instance]):

        thinking_mode = """You must think step by step and provide detailed thinking on the problem before giving the final answer.\nYou must put your thinking process between <think> and </think> tags and then output the final answer with a summary of your thinking process.\nIn your thinking process, this requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop a well-considered thinking process."""
        not_thinking_mode = """You are not required to have detailed thinking on the problem between <think> and </think> tags.\nYou can provide a direct answer to the question without detailed thinking.\nYou can still take steps to solve the problem, but you do not need to provide detailed thinking on the problem."""

        if requests:
            self.task = getattr(requests[0], "task_name", "unknown")

        print(f"[LLaDAEvalHarness] Generating with task: {self.task}")

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(
            self.llada_generate_until_tokenize,
            fn_kwargs={"tokenizer": self.tokenizer},
            batched=False,
            with_indices=True,
        )
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):

            if self.use_thinking:
                question_text = f"{thinking_mode}\n{elem['question_text']}"
            else:
                question_text = f"{not_thinking_mode}\n{elem['question_text']}"

            raw = question_text
            chat_history = [{"role":"user","content": raw}]
            prompt_str = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )

            prompt_ids = self.tokenizer(prompt_str)["input_ids"]
            prompt = torch.tensor([prompt_ids], device=self.device)

            generated_ids, _ = generate_with_dual_cache(
                self.model,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0.0,
                remasking="low_confidence",
                threshold=0.9,
                repetition_penalty=1.0,
            )

            gen_part = generated_ids[0, len(prompt_ids) :]
            text = self.tokenizer.decode(gen_part, skip_special_tokens=False)

            stop_token = "<|eot_id|>"
            if stop_token in text:
                text = text[: text.index(stop_token)]

            final_ids = self.tokenizer(text)["input_ids"]
            clean = self.tokenizer.decode(final_ids, skip_special_tokens=True)

            out.append(clean)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out
    
    def generate_until(self, requests: list[Instance]):

        thinking_mode = """You must think step by step and provide detailed thinking on the problem before giving the final answer.\nYou must put your thinking process between <think> and </think> tags and then output the final answer with a summary of your thinking process.\nIn your thinking process, this requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop a well-considered thinking process."""
        not_thinking_mode = """"""

        if requests:
            self.task = getattr(requests[0], "task_name", "unknown")

        print(f"[LLaDAEvalHarness] Generating with task: {self.task}")
        print(f"self.use_thinking = {self.use_thinking}")
        print(f"slef.repetition_penalty = {self.repetition_penalty}")

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(
            self.llada_generate_until_tokenize,
            fn_kwargs={"tokenizer": self.tokenizer},
            batched=False,
            with_indices=True,
        )
        ds = ds.with_format("torch")

        all_prompts: list[list[int]] = []
        raw_texts: list[str] 	 = []

        self.tokenizer.padding_side = "left"

        for elem in tqdm(ds, desc="Preparing prompts..."):
            if self.use_thinking:
                q = f"{thinking_mode}\n{elem['question_text']}"
            else:
                q = f"{not_thinking_mode}\n{elem['question_text']}"
            prompt_str = self.tokenizer.apply_chat_template(
                [{"role":"user","content": q}],
                tokenize=False,
                add_generation_prompt=True
            )
            ids = self.tokenizer(prompt_str)["input_ids"]
            all_prompts.append(ids)
            raw_texts.append(prompt_str)

        out = []
        for i in tqdm(range(0, len(all_prompts), self.generate_batch_size), desc="Generating..."):
            batch_texts = raw_texts[i : i + self.generate_batch_size]

            enc = self.tokenizer(
                batch_texts,
                padding=True, 	 	 	 	
                return_tensors="pt", 	 	 	
                add_special_tokens=False 	
            )
            input_ids 	 = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask", None)

            generated, _ = generate_with_dual_cache(
                self.model,
                input_ids,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0.0,
                remasking=self.remasking,
                threshold=0.9,
                repetition_penalty=self.repetition_penalty
            )

            if attention_mask is not None:
                prompt_lens = attention_mask.sum(dim=1)
            else:
                prompt_lens = torch.tensor([input_ids.shape[1]] * input_ids.shape[0],
                                           device=input_ids.device)

            for j, plen in enumerate(prompt_lens):
                start = plen.item()
                gen_ids = generated[j, start:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

                print(f"answer: \n {text}")

                if "<|eot_id|>" in text:
                    text = text[: text.index("<|eot_id|>")]

                final_ids = self.tokenizer(text)["input_ids"]
                clean 	  = self.tokenizer.decode(final_ids, skip_special_tokens=True)

                out.append(clean)

        return out

if __name__ == "__main__":
    set_seed(1234)
    
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process

    if is_main_process:
        args_str = " ".join(sys.argv)

        f = io.StringIO()
        with redirect_stdout(f):
            cli_evaluate()
        printed_output = f.getvalue()
        print(printed_output, flush=True)

        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "args": args_str,
            "output": printed_output,
        }

        os.makedirs("eval_logs", exist_ok=True)
        with open("eval_logs/eval_results_log.txt", "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    else:
        cli_evaluate()