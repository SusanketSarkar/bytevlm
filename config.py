"""All the knobs for the byte-VLM experiment, in one place."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ByteVLMConfig:
    # --- base LM ---
    model_name: str = "HuggingFaceTB/SmolLM2-135M"

    # --- image -> bytes ---
    image_size: int = 32           # resized to image_size x image_size
    grayscale: bool = True         # 1 channel vs 3 (RGB) -- controls sequence length
    img_start_token: str = "<img_start>"
    img_end_token: str = "<img_end>"
    pixel_token_prefix: str = "<px_"   # pixel-byte tokens are named <px_0> .. <px_255>

    # --- dataset ---
    dataset_kind: str = "caption"      # "caption" or "vqa"
    dataset_name: Optional[str] = None     # HF hub dataset name, e.g. "nlphuji/flickr30k"
    dataset_config: Optional[str] = None   # HF dataset config/subset name, if any
    dataset_split: str = "train"
    local_jsonl: Optional[str] = None      # alternative to dataset_name: local jsonl file
    image_col: str = "image"
    caption_col: str = "caption"
    question_col: str = "question"
    answer_col: str = "answer"
    max_examples: Optional[int] = None     # cap dataset size for quick smoke tests

    # --- sequence construction ---
    max_text_len: int = 64          # max tokens for prompt+answer combined (truncated)

    # --- training ---
    output_dir: str = "./runs/byte-vlm"
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    bf16: bool = True
    seed: int = 42
    dataloader_num_workers: int = 4

    @property
    def num_pixel_tokens(self) -> int:
        return 256  # one token per byte value, always -- resolution just changes seq length

    @property
    def image_token_count(self) -> int:
        channels = 1 if self.grayscale else 3
        return self.image_size * self.image_size * channels

    def special_tokens(self):
        """All new tokens to add to the base tokenizer's vocab."""
        px = [f"{self.pixel_token_prefix}{i}>" for i in range(self.num_pixel_tokens)]
        return [self.img_start_token, self.img_end_token] + px
