"""Wraps a plain HF causal LM (e.g. SmolLM2) and extends its vocabulary with
256 pixel-byte tokens + <img_start>/<img_end>. No architecture changes at all --
the model is still a vanilla decoder-only transformer; it just has more rows in its
embedding / output-projection matrices.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ByteVLMConfig


def load_model_and_tokenizer(cfg: ByteVLMConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    new_tokens = cfg.special_tokens()
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {added} new tokens (expected {len(new_tokens)}) to tokenizer.")

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # New rows are randomly initialized by resize_token_embeddings by default, which
    # is what we want: the model has to learn a meaning for each byte value from
    # scratch, purely from finetuning signal.

    return model, tokenizer
