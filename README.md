# byte-vlm

Experiment: can a *pure* language model (e.g. SmolLM2) become a VLM if we feed it
raw image bytes as extra "tokens" — no vision encoder, no patch embeddings, no CLIP —
and just finetune on captioning/VQA data?

## Idea

1. Resize an image to a small fixed resolution (e.g. 32x32 or 16x16, optionally grayscale)
   so the byte sequence stays short enough for the LM's context window.
2. Flatten the raw pixel bytes (values 0-255) into a sequence of integers.
3. Extend the LM's vocabulary with 256 new "pixel-byte" tokens (one per byte value) plus
   `<img_start>` / `<img_end>` sentinel tokens. Resize the model's input/output embeddings.
4. Build a sequence: `<bos> <img_start> [pixel byte tokens...] <img_end> [prompt text] <answer/caption text> <eos>`
5. Standard causal-LM finetuning, with the loss masked to only the answer/caption span
   (everything before it — including all the pixel tokens — is context, not something to
   be predicted).
6. If loss goes down and the model starts answering VQA / captioning questions correctly
   from nothing but raw bytes, that's evidence a decoder-only LM can learn to "read"
   images without any vision-specific inductive bias.

This is deliberately the simplest possible instantiation of the idea (bytes as
vocabulary, no byte-level convolution/patchification, no positional tricks). It's meant
as a baseline you can make more sophisticated later (e.g. byte-pair grouping, a small
conv stem over the bytes, 2D positional embeddings for pixel tokens, etc).

## Files

- `config.py` — all the knobs (model name, image resolution, grayscale, dataset, LR, ...)
- `image_bytes.py` — image -> fixed-length byte-token sequence
- `data.py` — dataset loading/formatting for captioning and VQA (HF `datasets`, or a local jsonl)
- `model.py` — wraps any HF `AutoModelForCausalLM` and extends its vocab with pixel-byte tokens
- `train.py` — the training entrypoint (HF `Trainer`-based)
- `infer.py` — quick manual test: give it an image + a question, see what it says
- `requirements.txt`

## Quickstart

```bash
pip install -r requirements.txt

python train.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --dataset_kind caption \
  --dataset_name nlphuji/flickr30k \
  --image_size 32 \
  --grayscale \
  --output_dir ./runs/smollm2-135m-byte-caption \
  --per_device_train_batch_size 8 \
  --num_train_epochs 1
```

For VQA:

```bash
python train.py \
  --model_name HuggingFaceTB/SmolLM2-135M \
  --dataset_kind vqa \
  --dataset_name HuggingFaceM4/VQAv2 \
  --image_size 32 \
  --grayscale \
  --output_dir ./runs/smollm2-135m-byte-vqa
```

Or point at your own local data (jsonl with `image_path` + `caption`, or
`image_path` + `question` + `answer`):

```bash
python train.py --dataset_kind caption --local_jsonl /path/to/captions.jsonl ...
```

Then try it:

```bash
python infer.py --checkpoint ./runs/smollm2-135m-byte-caption --image /path/to/img.jpg \
  --prompt "Describe this image."
```

## Sequence length math

Image tokens = `image_size * image_size * (1 if grayscale else 3)`. A SmolLM2-135M has a
2048/8192-token context depending on variant, so:

- 16x16 grayscale = 256 tokens
- 32x32 grayscale = 1024 tokens
- 32x32 RGB = 3072 tokens (needs a longer-context base model)
- 64x64 grayscale = 4096 tokens

Start small (16x16 or 32x32 grayscale) — the point of the experiment is to see whether
*any* signal gets through, not to get high-fidelity vision for free.
