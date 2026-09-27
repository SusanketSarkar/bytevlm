import argparse
import dataclasses

from transformers import Trainer, TrainingArguments, set_seed

from config import ByteVLMConfig
from data import ByteVLMDataset, make_collate_fn
from model import load_model_and_tokenizer


def parse_args() -> ByteVLMConfig:
    defaults = ByteVLMConfig()
    p = argparse.ArgumentParser(description="Finetune a byte-level pseudo-VLM.")
    for f in dataclasses.fields(defaults):
        default = getattr(defaults, f.name)
        if f.type == bool or isinstance(default, bool):
            p.add_argument(f"--{f.name}", action="store_true", default=default)
            p.add_argument(f"--no_{f.name}", dest=f.name, action="store_false")
        else:
            p.add_argument(f"--{f.name}", type=type(default) if default is not None else str,
                            default=default)
    args = p.parse_args()
    return ByteVLMConfig(**vars(args))


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    if not cfg.local_jsonl and not cfg.dataset_name:
        raise ValueError("Provide either --dataset_name (HF hub) or --local_jsonl.")

    print("Config:", cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)

    train_dataset = ByteVLMDataset(cfg, tokenizer)
    print(f"Loaded {len(train_dataset)} training examples "
          f"(image sequence length = {cfg.image_token_count} tokens).")

    collate_fn = make_collate_fn(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    trainer.train()

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Saved finetuned model + tokenizer to {cfg.output_dir}")


if __name__ == "__main__":
    main()
