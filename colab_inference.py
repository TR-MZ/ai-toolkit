"""
Google Colab inference script for ai-toolkit FLUX.2 Klein transparent image generation.

To use this on Colab:
1. Create a new Colab notebook
2. Copy the code below into cells
3. Replace PUBLIC_URL with the URL from run_with_public_url.py on your local machine

Or simply run:
    python colab_inference.py --url "https://xxxx.ngrok.io" --prompt "your prompt here"
"""

import requests
import json
import time
import argparse
from pathlib import Path

# Configuration
PUBLIC_URL = "https://YOUR_PUBLIC_URL_HERE.ngrok.io"  # Replace with output from run_with_public_url.py
OUTPUT_DIR = Path("generated_images")

def create_job(public_url: str, prompt: str, lora_scale: float = 1.0,
               steps: int = 20, width: int = 512, height: int = 512):
    """Create a new generation job via the web UI API."""

    # Get list of available jobs to create a new one
    try:
        response = requests.get(f"{public_url}/api/jobs")
        response.raise_for_status()
        print(f"✅ Connected to UI at {public_url}")
    except Exception as e:
        print(f"❌ Failed to connect to {public_url}")
        print(f"Error: {e}")
        print(f"\nMake sure you:")
        print(f"1. Started the UI with: python run_with_public_url.py")
        print(f"2. Replaced PUBLIC_URL with the ngrok URL it printed")
        return False

    # Create job config
    job_config = {
        "name": f"colab_gen_{int(time.time())}",
        "process": [
            {
                "type": "sd_trainer",
                "config": {
                    "model_config": {
                        "name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
                        "arch": "flux2_klein_4b",
                        "dtype": "bf16",
                        "vae_path": "/home/user/Documents/AlphaVAE/output/local_vae_training/checkpoint-7100",
                        "quantize": False,
                        "quantize_te": True,
                        "low_vram": True,
                    },
                    "sample_config": {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": steps,
                        "guidance_scale": 1.0,
                        "output_folder": "output/colab_generation",
                        "output_ext": "png",
                        "lora_path": "output/TransparentFluxLora_BG_BL_TO_FG_copy_copy_copy_copy_copy_copy/TransparentFluxLora_BG_BL_TO_FG_copy_copy_copy_copy_copy_copy.safetensors",
                        "lora_scale": lora_scale,
                    }
                }
            }
        ]
    }

    print(f"📝 Creating generation job for prompt: '{prompt}'")
    try:
        response = requests.post(
            f"{public_url}/api/jobs",
            json=job_config,
            timeout=30
        )
        response.raise_for_status()
        job = response.json()
        print(f"✅ Job created: {job.get('id')}")
        return job.get("id")
    except Exception as e:
        print(f"❌ Failed to create job: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Generate transparent images via remote ai-toolkit UI"
    )
    parser.add_argument(
        "--url",
        default=PUBLIC_URL,
        help="Public URL from run_with_public_url.py"
    )
    parser.add_argument(
        "--prompt",
        default="a transparent glass apple",
        help="Image generation prompt"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA strength (0.0 to 2.0)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height"
    )

    args = parser.parse_args()

    if args.url == PUBLIC_URL and "YOUR_PUBLIC_URL" in PUBLIC_URL:
        print("❌ Error: PUBLIC_URL not configured!")
        print("\nTo use this script:")
        print("1. On your local machine, run:")
        print("   python run_with_public_url.py")
        print("\n2. Copy the public URL it prints")
        print("\n3. Then run this script:")
        print(f"   python colab_inference.py --url 'https://xxxx.ngrok.io' --prompt 'your prompt'")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Create generation job
    job_id = create_job(
        args.url,
        args.prompt,
        lora_scale=args.lora_scale,
        steps=args.steps,
        width=args.width,
        height=args.height
    )

    if not job_id:
        return

    print(f"\n✨ Generation started!")
    print(f"Check the web UI at: {args.url}")
    print(f"Job ID: {job_id}")

if __name__ == "__main__":
    main()
