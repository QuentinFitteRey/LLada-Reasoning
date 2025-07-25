from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class PromptDataset(Dataset):
    """
    Final, corrected Dataset for the GRPO model.
    It applies the chat template and calculates prompt lengths in the collate_fn.
    """

    def __init__(self, dataset, tokenizer, max_len: int, strategy, input_template = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Store the raw dataset. Processing will happen "lazily" in the collate_fn.
        self.raw_dataset = dataset

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        # Return the raw data dictionary.
        # Assuming the key for the prompt is 'prompt'.
        return self.raw_dataset[idx]
    
    def collate_fn(self, batch):
        """
        This method now correctly applies the chat template before tokenizing.
        It also temporarily sets padding_side to "left" for generation without
        permanently changing the tokenizer's state.
        """
        # 1. Extract raw prompt strings from the batch
        prompt_list = [item['prompt'] for item in batch]

        # 2. Apply the chat template to each prompt
        formatted_prompts = []
        for p in prompt_list:
            messages = [{"role": "user", "content": p.strip()}]
            formatted_p = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            formatted_prompts.append(formatted_p)

        # 3. Store original padding side, set to 'left' for generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        # 4. Tokenize the list of *formatted* strings
        tokenized_batch = self.tokenizer(
            formatted_prompts,
            padding="longest",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # 5. Restore the original padding side
        self.tokenizer.padding_side = original_padding_side

        # 6. Calculate the length of each formatted prompt
        prompt_lens = tokenized_batch.attention_mask.sum(dim=1)
        
        # 7. Return the final tensor dictionary for the model
        return {
            "prompt_texts": formatted_prompts,
            "prompt_ids": tokenized_batch['input_ids'],
            "prompt_mask": tokenized_batch['attention_mask'],
            "prompt_lens": prompt_lens
        }