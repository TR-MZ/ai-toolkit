import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import AutoencoderKL
try:
    from diffusers import AutoencoderKLFlux2
except ImportError:
    AutoencoderKLFlux2 = None

def load_vae(path, device):
    print(f"Loading VAE from {path}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    try:
        if AutoencoderKLFlux2 is not None:
            vae = AutoencoderKLFlux2.from_pretrained(path, torch_dtype=dtype)
        else:
            raise ImportError("AutoencoderKLFlux2 not available")
    except Exception as e:
        print(f"Could not load as AutoencoderKLFlux2 ({e}), trying AutoencoderKL")
        vae = AutoencoderKL.from_pretrained(path, torch_dtype=dtype)
    vae = vae.to(device)
    vae.eval()
    return vae

def process_image(vae, image_tensor):
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
        reconstructed = vae.decode(latent).sample
    return reconstructed

def tensor_to_pil(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    rgb = (arr[:, :, :3] + 1.0) / 2.0
    alpha = (arr[:, :, 3:4] + 1.0) / 2.0
    rgba = np.concatenate([rgb, alpha], axis=2)
    rgba = np.clip(rgba, 0, 1)
    rgba = (rgba * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")

def composite_on_bg(img_rgba, bg_color):
    bg = Image.new("RGBA", img_rgba.size, bg_color + (255,))
    bg.paste(img_rgba, mask=img_rgba.split()[3])
    return bg.convert("RGB")

def add_label(img, text):
    draw = ImageDraw.Draw(img)
    # try to load a font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), text, fill=(255, 0, 0), font=font)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input PNG (RGBA)")
    parser.add_argument("--model1", type=str, default="models/finetune_VAE_frozen_1500steps", help="Path to Frozen VAE 1500")
    parser.add_argument("--model2", type=str, default="models/finetune_VAE", help="Path to Regular VAE")
    parser.add_argument("--output", type=str, default="comparison_output.png", help="Output comparison image path")
    parser.add_argument("--size", type=int, default=512, help="Resize input to this size")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load input image
    img = Image.open(args.image).convert("RGBA")
    img = img.resize((args.size, args.size), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16 if device == "cuda" else torch.float32)

    # Load models
    vae1 = load_vae(args.model1, device)
    vae2 = load_vae(args.model2, device)

    # Process
    print("Processing with Model 1 (Frozen)...")
    recon1 = process_image(vae1, tensor)
    pil1 = tensor_to_pil(recon1)

    print("Processing with Model 2 (Regular)...")
    recon2 = process_image(vae2, tensor)
    pil2 = tensor_to_pil(recon2)

    # Create comparison
    # We will show them on checkerboard background to see alpha
    def make_checkerboard(size, tile=32):
        w, h = size
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                color = 200 if ((x // tile) + (y // tile)) % 2 == 0 else 128
                arr[y:y+tile, x:x+tile] = color
        return Image.fromarray(arr, mode="RGB")
        
    checker = make_checkerboard((args.size, args.size))
    
    # Original
    orig_comp = checker.copy().convert("RGBA")
    orig_comp.paste(img, mask=img.split()[3])
    orig_comp = orig_comp.convert("RGB")
    add_label(orig_comp, "Original")


    # M1
    m1_comp = checker.copy().convert("RGBA")
    m1_comp.paste(pil1, mask=pil1.split()[3])
    m1_comp = m1_comp.convert("RGB")
    add_label(m1_comp, f"Frozen 1500")

    # M2
    m2_comp = checker.copy().convert("RGBA")
    m2_comp.paste(pil2, mask=pil2.split()[3])
    m2_comp = m2_comp.convert("RGB")
    add_label(m2_comp, f"Regular Finetune")

    # Stitch
    final_w = args.size * 3
    final_h = args.size
    final_img = Image.new("RGB", (final_w, final_h))
    final_img.paste(orig_comp, (0, 0))
    final_img.paste(m1_comp, (args.size, 0))
    final_img.paste(m2_comp, (args.size * 2, 0))
    
    final_img.save(args.output)
    print(f"Saved comparison to {args.output}")

if __name__ == "__main__":
    main()
