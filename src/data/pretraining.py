# import torch
from torch.utils.data import Dataset
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        print(f"[DATASET ITEM] idx={idx} len={len(item['input_ids'])}", flush=True)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _chunk_raw_text(self, raw_text):
        # Start of debug edit
        chunks = []
        buffer_ids = []
        for line in raw_text:
            line_ids = self.tokenizer(line, add_special_tokens=False)["input_ids"]
            buffer_ids.extend(line_ids)
            while len(buffer_ids) >= self.max_length:
                chunk_ids = buffer_ids[: self.max_length]
                chunks.append(self.tokenizer.decode(chunk_ids))
                buffer_ids = buffer_ids[self.max_length :]
        if buffer_ids:
            chunks.append(self.tokenizer.decode(buffer_ids))
        return chunks

    # End
    #        raw_text = "\n\n".join(raw_text)
    #        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
    #            "input_ids"
    #        ]
    #        num_chunks = len(full_token_sequence) // self.max_length + 1
    #        chunks = []
    #        for i in range(num_chunks):
    #            chunks.append(
    #                self.tokenizer.decode(
    #                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
    #                )
    #            )
    #        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )
