"""Turn a PIL image into a fixed-length sequence of pixel-byte *token strings*.

Each pixel byte value (0-255) maps 1:1 to a vocab token "<px_{value}>", so the model
sees the raw bytes of the (downsized) image as ordinary discrete tokens, with zero
vision-specific structure imposed beyond "resize then flatten row-major".
"""
from typing import List
from PIL import Image

from config import ByteVLMConfig


def image_to_pixel_tokens(image: Image.Image, cfg: ByteVLMConfig) -> List[str]:
    img = image.convert("L" if cfg.grayscale else "RGB")
    img = img.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)

    if cfg.grayscale:
        flat = list(img.getdata())  # ints 0-255
    else:
        flat = [v for pixel in img.getdata() for v in pixel]  # (r,g,b) tuples -> flat ints

    assert len(flat) == cfg.image_token_count, (
        f"expected {cfg.image_token_count} bytes, got {len(flat)}"
    )
    return [f"{cfg.pixel_token_prefix}{b}>" for b in flat]


def wrap_with_sentinels(pixel_tokens: List[str], cfg: ByteVLMConfig) -> List[str]:
    return [cfg.img_start_token] + pixel_tokens + [cfg.img_end_token]
