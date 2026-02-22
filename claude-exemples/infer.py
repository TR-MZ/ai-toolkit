#!/usr/bin/env python3
"""
Inference script for FLUX.2-klein + custom RGBA VAE + ai-toolkit LoRA.

Usage:
  python claude-exemples/infer.py
  python claude-exemples/infer.py --prompt "a transparent glass skull" --steps 20
  python claude-exemples/infer.py --lora path/to/lora.safetensors --lora_scale 0.9
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile

# ── Config ─────────────────────────────────────────────────────────────────────
VAE_PATH   = "/home/user/Documents/AlphaVAE/output/local_vae_training/checkpoint-7100"
LORA_PATH  = "output/TransparentFluxLora_BG_BL_TO_FG_copy_copy_copy_copy_copy_copy/TransparentFluxLora_BG_BL_TO_FG_copy_copy_copy_copy_copy_copy.safetensors"
PROMPT     = "a transparent glass apple"
WIDTH      = 512
HEIGHT     = 512
STEPS      = 20
GUIDANCE   = 1.0
LORA_SCALE = 1.0
SEED       = 42
OUTPUT_DIR = Path("output/infer")
# ───────────────────────────────────────────────────────────────────────────────

# add repo root to path so toolkit imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fit_to_canvas(img: Image.Image, size: int = 512) -> Image.Image:
    """Scale image to fit within size×size, center on a transparent RGBA canvas."""
    img = img.convert("RGBA")
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2
    canvas.paste(img, (offset_x, offset_y), mask=img.split()[3])
    return canvas


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


def apply_lora_to_transformer(transformer, lora_path: str, scale: float = 1.0):
    """Merge LoRA weights into the transformer in-place.

    ai-toolkit saves keys as: diffusion_model.<block>.<layer>.lora_A.weight
                                                                   lora_B.weight
    """
    from safetensors.torch import load_file
    sd = load_file(lora_path)

    # Group into (lora_A, lora_B, alpha) triplets keyed by base module path
    lora_A, lora_B, alphas = {}, {}, {}
    for k, v in sd.items():
        # strip "diffusion_model." prefix → transformer internal path
        base = k.replace("diffusion_model.", "", 1)
        if base.endswith(".lora_A.weight"):
            key = base[: -len(".lora_A.weight")]
            lora_A[key] = v
        elif base.endswith(".lora_B.weight"):
            key = base[: -len(".lora_B.weight")]
            lora_B[key] = v
        elif base.endswith(".alpha"):
            key = base[: -len(".alpha")]
            alphas[key] = v.item()

    merged = 0
    for key in lora_A:
        if key not in lora_B:
            print(f"  [warn] missing lora_B for {key}, skipping")
            continue
        A = lora_A[key].float()
        B = lora_B[key].float()
        dim = A.shape[0]
        alpha = alphas.get(key, dim)
        delta = (B @ A) * (alpha / dim) * scale  # (out, in)

        # navigate to the target module and update its weight
        parts = key.split(".")
        module = transformer
        for part in parts:
            module = getattr(module, part)
        # module is now the Linear layer; update its weight in-place
        module.weight.data += delta.to(module.weight.dtype).to(module.weight.device)
        merged += 1

    print(f"  Merged {merged} LoRA modules into transformer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae",        default=VAE_PATH)
    parser.add_argument("--lora",       default=LORA_PATH)
    parser.add_argument("--prompt",     default=PROMPT)
    parser.add_argument("--steps",      type=int,   default=STEPS)
    parser.add_argument("--guidance",   type=float, default=GUIDANCE)
    parser.add_argument("--width",      type=int,   default=WIDTH)
    parser.add_argument("--height",     type=int,   default=HEIGHT)
    parser.add_argument("--seed",       type=int,   default=SEED)
    parser.add_argument("--lora_scale", type=float, default=LORA_SCALE)
    parser.add_argument("--ctrl_img_1", default=None, help="Path to first control image")
    parser.add_argument("--ctrl_img_2", default=None, help="Path to second control image")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    print(f"Device: {device}  |  dtype: {dtype}")

    # ── load model via toolkit ──────────────────────────────────────────────────
    from toolkit.config_modules import ModelConfig
    from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein4BModel

    model_config = ModelConfig(
        name_or_path="black-forest-labs/FLUX.2-klein-base-4B",
        arch="flux2_klein_4b",
        dtype=str(dtype).replace("torch.", ""),
        vae_path=args.vae,
        quantize=False,
        quantize_te=True,
        low_vram=True,
    )

    print("Loading model...")
    sd = Flux2Klein4BModel(
        device=device,
        model_config=model_config,
        dtype="bf16",
    )
    sd.load_model()

    # ── apply LoRA ──────────────────────────────────────────────────────────────
    if args.lora and os.path.exists(args.lora):
        print(f"Applying LoRA: {args.lora}  (scale={args.lora_scale})")
        apply_lora_to_transformer(sd.transformer, args.lora, scale=args.lora_scale)
    elif args.lora:
        print(f"[warn] LoRA not found: {args.lora}")

    # ── preprocess control images (fit to 512×512 canvas) ───────────────────────
    _tmp_dir = None
    ctrl_img_1 = args.ctrl_img_1
    ctrl_img_2 = args.ctrl_img_2

    if ctrl_img_1 or ctrl_img_2:
        _tmp_dir = tempfile.mkdtemp(prefix="infer_ctrl_")
        if ctrl_img_1 and os.path.exists(ctrl_img_1):
            fitted = fit_to_canvas(Image.open(ctrl_img_1), size=args.width)
            ctrl_img_1 = os.path.join(_tmp_dir, "ctrl1.png")
            fitted.save(ctrl_img_1)
            print(f"  ctrl_img_1 fitted to {args.width}×{args.height}: {ctrl_img_1}")
        if ctrl_img_2 and os.path.exists(ctrl_img_2):
            fitted = fit_to_canvas(Image.open(ctrl_img_2), size=args.width)
            ctrl_img_2 = os.path.join(_tmp_dir, "ctrl2.png")
            fitted.save(ctrl_img_2)
            print(f"  ctrl_img_2 fitted to {args.width}×{args.height}: {ctrl_img_2}")

    # ── generate ────────────────────────────────────────────────────────────────
    from toolkit.config_modules import GenerateImageConfig

    gen_config = GenerateImageConfig(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        seed=args.seed,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        output_folder=str(OUTPUT_DIR),
        output_ext="png",
        ctrl_img_1=ctrl_img_1,
        ctrl_img_2=ctrl_img_2,
    )

    print(f"Generating: '{args.prompt}'")
    sd.generate_images([gen_config])

    # the image is already saved by generate_images, but also show the compare
    # cleanup temp control image files
    if _tmp_dir:
        import shutil
        shutil.rmtree(_tmp_dir, ignore_errors=True)

    saved = list(OUTPUT_DIR.glob("*.png"))
    saved.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    # skip _compare files from previous runs
    saved = [p for p in saved if "_compare" not in p.name]
    if saved:
        rgba = Image.open(saved[0])
        # fit output to 512×512 canvas before compositing
        if rgba.mode == "RGBA":
            rgba = fit_to_canvas(rgba, size=args.width)
            rgba.save(saved[0])
        if rgba.mode == "RGBA":
            w, h = rgba.size
            white   = Image.new("RGB", (w, h), (255, 255, 255))
            black   = Image.new("RGB", (w, h), (0,   0,   0))
            checker = checkerboard((h, w))
            on_white   = composite(rgba, white.convert("RGBA"))
            on_black   = composite(rgba, black.convert("RGBA"))
            on_checker = composite(rgba, checker.convert("RGBA"))
            gap      = 8
            canvas   = Image.new("RGB", (w * 3 + gap * 2, h), (30, 30, 30))
            canvas.paste(on_white,   (0,          0))
            canvas.paste(on_black,   (w + gap,    0))
            canvas.paste(on_checker, (w * 2 + gap * 2, 0))
            compare_path = saved[0].with_name(saved[0].stem + "_compare.png")
            canvas.save(compare_path)
            print(f"\nSaved:")
            print(f"  RGBA:    {saved[0]}")
            print(f"  Compare: {compare_path}")
        else:
            print(f"\nSaved: {saved[0]}")


if __name__ == "__main__":
    main()
