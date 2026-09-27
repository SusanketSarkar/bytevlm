"""Loads captioning/VQA data and turns each example into
    <bos> <img_start> [pixel tokens] <img_end> [prompt] [answer] <eos>
with `labels` masked (-100) everywhere except the answer/caption span, so the loss
only ever asks the model to predict text conditioned on the raw image bytes.
"""
import json
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from config import ByteVLMConfig
from image_bytes import image_to_pixel_tokens, wrap_with_sentinels


class ByteVLMDataset(Dataset):
    def __init__(self, cfg: ByteVLMConfig, tokenizer):
        self.cfg = cfg
        self.tok = tokenizer
        self.examples = self._load_examples()

    # ---------- loading ----------

    def _load_examples(self):
        if self.cfg.local_jsonl:
            examples = []
            with open(self.cfg.local_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
        else:
            from datasets import load_dataset
            ds = load_dataset(
                self.cfg.dataset_name,
                self.cfg.dataset_config,
                split=self.cfg.dataset_split,
            )
            examples = ds

        if self.cfg.max_examples is not None:
            examples = examples[: self.cfg.max_examples] if isinstance(examples, list) \
                else examples.select(range(min(self.cfg.max_examples, len(examples))))
        return examples

    def __len__(self):
        return len(self.examples)

    # ---------- formatting ----------

    def _get_image(self, ex) -> Image.Image:
        img = ex[self.cfg.image_col]
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, dict) and "path" in img:
            return Image.open(img["path"])
        if isinstance(img, str):
            return Image.open(img)
        raise ValueError(f"Don't know how to load image from: {type(img)}")

    def _get_prompt_and_answer(self, ex):
        if self.cfg.dataset_kind == "caption":
            caption = ex[self.cfg.caption_col]
            if isinstance(caption, list):  # some caption datasets give multiple refs
                caption = caption[0]
            prompt = "Describe this image."
            answer = caption
        elif self.cfg.dataset_kind == "vqa":
            question = ex[self.cfg.question_col]
            answer = ex[self.cfg.answer_col]
            if isinstance(answer, list):  # VQAv2-style multiple annotator answers
                answer = answer[0]
            prompt = question
        else:
            raise ValueError(f"Unknown dataset_kind: {self.cfg.dataset_kind}")
        return prompt, str(answer)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        cfg = self.cfg

        image = self._get_image(ex)
        pixel_tokens = wrap_with_sentinels(image_to_pixel_tokens(image, cfg), cfg)
        img_ids = self.tok.convert_tokens_to_ids(pixel_tokens)

        prompt, answer = self._get_prompt_and_answer(ex)
        prompt_ids = self.tok.encode(f"\nQuestion: {prompt}\nAnswer:", add_special_tokens=False)
        answer_ids = self.tok.encode(" " + answer.strip(), add_special_tokens=False)

        # truncate the text part (never the image bytes) if it's too long
        max_text = cfg.max_text_len
        if len(prompt_ids) + len(answer_ids) > max_text:
            keep_answer = min(len(answer_ids), max_text // 2)
            answer_ids = answer_ids[:keep_answer]
            prompt_ids = prompt_ids[: max_text - keep_answer]

        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        eos = [self.tok.eos_token_id] if self.tok.eos_token_id is not None else []

        input_ids = bos + img_ids + prompt_ids + answer_ids + eos
        # labels: -100 (ignored) for everything except the answer + eos
        labels = (
            [-100] * (len(bos) + len(img_ids) + len(prompt_ids))
            + answer_ids
            + eos
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_collate_fn(pad_token_id: int):
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = b["input_ids"]
            labels[i, :n] = b["labels"]
            attention_mask[i, :n] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate
