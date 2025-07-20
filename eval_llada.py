'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
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
from new_generation_quentin import generate_with_dual_cache
from generate import generate
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
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=128,
        remasking='low_confidence',
        device="cuda",
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.load_lora = load_lora
        self.task = os.environ.get("EVAL_TASKS", None)

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
            device=device,
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

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

    # def apply_chat_template(
    #     self,
    #     chat_history: list[dict[str, str]],
    #     use_thinking: bool = True
    # ) -> str:
    #     """
    #     If use_thinking is True, we prefix the user message with the
    #     thinking_mode instructions; otherwise with not_thinking_mode.
    #     """

    #     thinking_mode = """You must think step by step and provide detailed thinking on the problem before giving the final answer.\nYou must put your thinking process between <think> and </think> tags and then output the final answer with a summary of your thinking process.\nIn your thinking process, this requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop a well-considered thinking process."""
    #     not_thinking_mode = """You are not required to have detailed thinking on the problem between <think> and </think> tags.\nYou can provide a direct answer to the question without detailed thinking.\nYou can still take steps to solve the problem, but you do not need to provide detailed thinking on the problem."""

    #     # grab the last user message
    #     user_msg = chat_history[-1]["content"]

    #     # choose the appropriate instruction prefix
    #     prefix = thinking_mode if use_thinking else not_thinking_mode

    #     # build and return the full prompt
    #     return (
    #         "<BOS>"
    #         "<start_id>user<end_id>\n"
    #         f"{prefix}\n"
    #         f"{user_msg}"
    #         "<eot_id>"
    #         "<start_id>assistant<end_id>\n"
    #     )

    def llada_generate_until_tokenize(self, e, idx, tokenizer):
        return {
            "question": tokenizer(e["question"])["input_ids"],
            "question_text": e["question"],
            "until": e["until"],
        }

    def generate_until(self, requests: list[Instance]):

        thinking_mode = """You must think step by step and provide detailed thinking on the problem before giving the final answer.\nYou must put your thinking process between <think> and </think> tags and then output the final answer with a summary of your thinking process.\nIn your thinking process, this requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop a well-considered thinking process."""
        not_thinking_mode = """You are not required to have detailed thinking on the problem between <think> and </think> tags.\nYou can provide a direct answer to the question without detailed thinking.\nYou can still take steps to solve the problem, but you do not need to provide detailed thinking on the problem."""
        use_thinking = True

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

            # 0) Concatenate the thinking mode or not thinking mode with the question_text
            if use_thinking:
                question_text = f"{thinking_mode}\n{elem['question_text']}"
            else:
                question_text = f"{not_thinking_mode}\n{elem['question_text']}"

            # 1) wrap the question in a chat template
            raw = question_text
            chat_history = [{"role":"user","content": raw}]
            prompt_str = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )

            # 2) tokenize
            prompt_ids = self.tokenizer(prompt_str)["input_ids"]
            prompt = torch.tensor([prompt_ids], device=self.device)

            # 3) generate with dual cache
            generated_ids, _ = generate_with_dual_cache(
                self.model,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0.0,
                remasking="low_confidence",
                threshold=0.8
            )

            # print(f"--- Debug Info for Request ---")
            # print(f"Prompt shape: {prompt.shape}")
            # print(f"Generated output tensor shape (from `generate`): {generated_answer.shape}")
            # print(f"Expected generated part start index (prompt.shape[1]): {prompt.shape[1]}")

            # # Check if the slice is empty
            # if generated_answer.shape[1] <= prompt.shape[1]:
            #     print("WARNING: Generated output tensor is not longer than the prompt! No new tokens were generated.")

            # 4) strip off the prompt tokens and decode
            gen_part = generated_ids[0, len(prompt_ids) :]
            text = self.tokenizer.decode(gen_part, skip_special_tokens=False)

            # 5) stop‐token trimming
            stop_token = "<|eot_id|>"
            if stop_token in text:
                text = text[: text.index(stop_token)]

            # 6) remove any stray special tokens
            final_ids = self.tokenizer(text)["input_ids"]
            clean = self.tokenizer.decode(final_ids, skip_special_tokens=True)

            # print("----- PROMPT BEGIN -----", file=sys.stderr)
            # print(prompt_str, file=sys.stderr)
            # print("------ PROMPT END ------\n", file=sys.stderr)
            # print("----- GENERATED BEGIN ----", file=sys.stderr)
            # print(text, file=sys.stderr)
            # print("------ GENERATED END -----\n", file=sys.stderr)
            # print("----- ANSWER BEGIN ----", file=sys.stderr)
            # print(clean, file=sys.stderr)
            # print("------ ANSWER END -----", file=sys.stderr)

            if self.task.startswith("mmlu"):
                # --- STAGE 1: Strict / structured matches ---
                primary_matches = re.findall(
                    r"(?:The (?:best )?answer is\s*\\boxed\{([A-Z])\}|"        # The best answer is \boxed{B}
                    r"\\boxed\{([A-Z])\}|"                                     # \boxed{B}
                    r"The (?:best )?answer is\s*[\(\[]?([A-Z])[\)\]]?|"        # The best answer is B
                    r"Answer[:\s]+([A-Z]))",                                   # Answer: B
                    clean,
                    re.IGNORECASE
                )
                flat_primary = [g for match in primary_matches for g in match if g]

                if flat_primary:
                    clean = flat_primary[-1].strip().upper()
                    # print(f"[LLaDAEvalHarness] Cleaned answer (primary match): {clean}", file=sys.stderr)
                else:
                    # --- STAGE 2: Heuristic / fallback patterns ---
                    secondary_matches = re.findall(
                        r"(?:Option\s+([A-Z])|"                                         # Option B
                        r"I (?:choose|pick|conclude with)\s+([A-Z])|"                   # I choose A
                        r"The correct answer is\s*[\(\[]?([A-Z])[\)\]]?|"               # The correct answer is (B)
                        r"Therefore, the correct answer is\s*[\(\[]?([A-Z])[\)\]]?|"    # Therefore, the correct answer is (A)
                        r"Thus, the answer is\s*([A-Z])|"                               # Thus, the answer is B
                        r"Hence,.*?\s([A-Z])|"                                          # Hence ... Z
                        r"^Answer\s*[\r\n]+([A-Z])|"                                    # Answer\nB
                        r"\*\*Answer:\s*\(([A-Z])\)\*\*|"                               # **Answer: (C)**
                        r"\*\*Answer:\s*\(([A-Z])\)|"                                   # **Answer:** (C)
                        r"\*\*Answer:\*\*\s*\[([A-Z])\]|"                               # **Answer:** [D]
                        r"^([A-Z])$"                                                    # Capital letter alone on a line
                        r")",                                                           # <-- THIS closes the outer non-capturing group
                        clean,
                        re.IGNORECASE | re.MULTILINE,
                    )
                    flat_secondary = [g for match in secondary_matches for g in match if g]

                    if flat_secondary:
                        clean = flat_secondary[-1].strip().upper()
                        # print(f"[LLaDAEvalHarness] Cleaned answer (secondary heuristic): {clean}", file=sys.stderr)
                    else:
                        print(f"[Warning] No valid A-Z answer found in generated text for MMLU task:\n{clean}", file=sys.stderr)

            elif self.task.startswith("gsm8k"):
                # --- STAGE 1: Strict pattern match ---
                match = re.search(
                    r"(?:####\s*|"
                    r"\\boxed\s*\{\s*|"
                    r"\\boxed\{|\bAnswer[:\s]*|"
                    r"The final answer is\s*\\boxed\{?|"
                    r"The answer is\s*\\boxed\{?)"
                    r"(-?\d{1,3}(?:,\d{3})*|\d+)(?:\s*\w+)?\}?",
                    clean,
                    re.IGNORECASE
                )

                if match:
                    extracted_answer = match.group(1).replace(",", "").strip()
                    answer_end_idx = match.end()
                    clean = clean[:answer_end_idx].rstrip() + f"\n#### {extracted_answer}"

                else:
                    # --- STAGE 2: Heuristic fallback for implicit answers ---
                    fallback_matches = re.findall(
                        r"(?:\*\*)?(?:takes|is|equals|costs|spent|was|be|add up to|total(?:s)?(?: up to)?|"
                        r"amount(?:s)? to|the answer is|answer[:\s]*|summary[:\s]*|=)(?:\*\*|:)?\s*"
                        r"\**\$?(-?\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\**"
                        r"(?:\s*(?:minutes?|seconds?|hours?|dollars?|cents?|units?|miles?|kilometers?))?",
                        clean,
                        re.IGNORECASE
                    )

                    if fallback_matches:
                        extracted_answer = fallback_matches[-1].replace(",", "").strip()
                        clean = clean.strip() + f"\n#### {extracted_answer}"
                    else:
                        print(f"[Warning] No numeric answer found in GSM8K output:\n{clean}", file=sys.stderr)

            elif self.task == "hendrycks_math": # WIP: Very bad rn
                # find all $…$ spans
                dollars = [i for i,ch in enumerate(text) if ch=='$']
                if len(dollars)>=2:
                    ans = text[dollars[0]+1 : dollars[-1]]
                else:
                    ans = text
                # strip any “\boxed{…}”
                ans = re.sub(r".*\\boxed\{(.*)\}.*", r"\1", ans, flags=re.DOTALL)
                clean = ans.strip()

            out.append(clean)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

if __name__ == "__main__":
    set_seed(1234)
    
    accelerator = Accelerator()
    is_main_process = accelerator.is_main_process

    if is_main_process:
        # Capture CLI args
        args_str = " ".join(sys.argv)

        # Redirect stdout to capture printed output
        f = io.StringIO()
        with redirect_stdout(f):
            cli_evaluate()
        printed_output = f.getvalue()
        print(printed_output, flush=True)

        # Construct log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "args": args_str,
            "output": printed_output,
        }

        # Append to a file
        os.makedirs("eval_logs", exist_ok=True)
        with open("eval_logs/eval_results_log.txt", "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")
    else:
        cli_evaluate()
    