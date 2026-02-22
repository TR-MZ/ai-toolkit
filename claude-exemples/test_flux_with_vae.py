#!/usr/bin/env python3
"""
Test a FLUX.2-klein LoRA checkpoint by generating a transparent image.
Output is saved as both a PNG (RGBA) and composited on white/black/checker.

Usage:
  python scripts/test_lora_checkpoint.py
  python scripts/test_lora_checkpoint.py --checkpoint output/flux2_klein_lora_alpha_v2/checkpoint-1000
  python scripts/test_lora_checkpoint.py --prompt "a transparent glass skull"

Mac/MPS compatible — auto-detects cuda > mps > cpu.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "black-forest-labs/FLUX.2-klein-base-4B"
VAE_PATH    = "/home/user/Documents/AlphaVAE/output/local_vae_training/checkpoint-7100"
CHECKPOINT  = "output/ostris_lora/TransparentFluxLora_Tra_ds_000002500.safetensors"
PROMPT      = "a transparent glass apple"
WIDTH       = 512
HEIGHT      = 512
STEPS       = 4
GUIDANCE    = 1.0
LORA_SCALE  = 1.0
OUTPUT_DIR  = Path("output/lora_tests")
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_dtype(device):
    # bfloat16 is well supported on cuda and mps (M2+); fall back to float16 on cpu
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16

def checkerboard(size, tile=32):
    H, W = size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            c = 176 if (x // tile + y // tile) % 2 == 0 else 232
            img[y:y+tile, x:x+tile] = c
    return Image.fromarray(img)

def composite(rgba: Image.Image, bg: Image.Image) -> Image.Image:
    out = bg.copy().convert("RGBA")
    out.paste(rgba, mask=rgba.split()[3])
    return out.convert("RGB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--prompt",     default=PROMPT)
    parser.add_argument("--steps",      type=int,   default=STEPS)
    parser.add_argument("--guidance",   type=float, default=GUIDANCE)
    parser.add_argument("--width",      type=int,   default=WIDTH)
    parser.add_argument("--height",     type=int,   default=HEIGHT)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--lora_scale", type=float, default=LORA_SCALE)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    dtype  = get_dtype(device)
    print(f"Device: {device}  |  dtype: {dtype}")

    from diffusers import AutoencoderKLFlux2, Flux2Pipeline
    from safetensors.torch import load_file
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    # Load custom RGBA VAE
    print(f"Loading RGBA VAE from {VAE_PATH}...")
    vae = AutoencoderKLFlux2.from_pretrained(VAE_PATH, torch_dtype=dtype)

    print(f"Loading pipeline from {BASE_MODEL}...")
    pipe = Flux2Pipeline.from_pretrained(
        BASE_MODEL,
        vae=vae,
        torch_dtype=dtype,
    )

    # Load LoRA weights — ai-toolkit saves with "diffusion_model." prefix, convert to "transformer."
    lora_path = Path(args.checkpoint)
    if lora_path.is_dir():
        lora_path = lora_path / "pytorch_lora_weights.safetensors"
    print(f"Loading LoRA from {lora_path} (scale={args.lora_scale})...")
    raw_sd = load_file(str(lora_path))
    # Rename keys: diffusion_model.X -> transformer.X
    lora_sd = {}
    for k, v in raw_sd.items():
        new_k = k.replace("diffusion_model.", "transformer.", 1)
        lora_sd[new_k] = v
    pipe.load_lora_weights(lora_sd)
    pipe.fuse_lora(lora_scale=args.lora_scale)

    # cpu_offload on CUDA, direct .to() on mps/cpu
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    print(f"Generating: '{args.prompt}'")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        output_type="latent",
    )

    # Decode through RGBA VAE — explicitly move to device since cpu_offload may have moved it
    if device.type == "cuda":
        pipe.vae.to(device)
    latents = result.images.to(device=device, dtype=dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(latents).sample
    if device.type == "cuda":
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded.clamp(0, 1)
    img_np = (decoded[0].permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
    rgba = Image.fromarray(img_np, mode="RGBA")

    # Save outputs
    step = Path(args.checkpoint).stem  # e.g. "checkpoint-500" or lora filename without ext
    slug = args.prompt[:40].replace(" ", "_").replace("/", "-")

    rgba.save(OUTPUT_DIR / f"{step}_{slug}_rgba.png")

    w, h = rgba.size
    white   = Image.new("RGB", (w, h), (255, 255, 255))
    black   = Image.new("RGB", (w, h), (0,   0,   0))
    checker = checkerboard((h, w))

    on_white   = composite(rgba, white.convert("RGBA"))
    on_black   = composite(rgba, black.convert("RGBA"))
    on_checker = composite(rgba, checker.convert("RGBA"))

    # Side-by-side comparison
    gap = 8
    total_w = w * 3 + gap * 2
    canvas = Image.new("RGB", (total_w, h), (30, 30, 30))
    canvas.paste(on_white,   (0,          0))
    canvas.paste(on_black,   (w + gap,    0))
    canvas.paste(on_checker, (w*2 + gap*2, 0))
    out_path = OUTPUT_DIR / f"{step}_{slug}_compare.png"
    canvas.save(out_path)

    print(f"\nSaved:")
    print(f"  RGBA:    {OUTPUT_DIR}/{step}_{slug}_rgba.png")
    print(f"  Compare: {out_path}")

if __name__ == "__main__":
    main()
