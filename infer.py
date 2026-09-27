import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ByteVLMConfig
from image_bytes import image_to_pixel_tokens, wrap_with_sentinels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="dir saved by train.py")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe this image.")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--grayscale", action="store_true", default=True)
    p.add_argument("--max_new_tokens", type=int, default=40)
    args = p.parse_args()

    cfg = ByteVLMConfig(image_size=args.image_size, grayscale=args.grayscale)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    model.eval()

    image = Image.open(args.image)
    pixel_tokens = wrap_with_sentinels(image_to_pixel_tokens(image, cfg), cfg)
    img_ids = tokenizer.convert_tokens_to_ids(pixel_tokens)

    prompt_ids = tokenizer.encode(f"\nQuestion: {args.prompt}\nAnswer:", add_special_tokens=False)
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []

    input_ids = torch.tensor([bos + img_ids + prompt_ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated = out[0][input_ids.shape[1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
