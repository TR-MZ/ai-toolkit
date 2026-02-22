# 🌐 Access ai-toolkit from Google Colab (or anywhere)

This guide shows how to run your ai-toolkit web UI publicly so you can access it from Google Colab, your phone, or any device with internet.

## Quick Setup (5 minutes)

### 1. Install ngrok on your local machine

```bash
pip install pyngrok
```

### 2. Start the public UI server

On your **local machine** (where GPU/VRAM is available):

```bash
python run_with_public_url.py
```

You'll see output like:

```
✅ Public tunnel created!

============================================================
🔗 PUBLIC URL (use this to access from Colab):
   https://xxxx-xxxx-xxxx.ngrok.io
============================================================

Local:  http://localhost:8675
Public: https://xxxx-xxxx-xxxx.ngrok.io

Press Ctrl+C to stop
```

**Copy the public URL** (the one starting with `https://`)

### 3. Use from Colab (or anywhere)

#### Option A: Web UI (easiest)
Simply paste the public URL into your browser. You can now:
- Create training jobs
- View the web UI
- Monitor training
- Generate images with LoRA

#### Option B: Python/Colab Script
Use `colab_inference.py` to generate images programmatically:

```bash
python colab_inference.py \
  --url "https://xxxx-xxxx-xxxx.ngrok.io" \
  --prompt "a transparent glass skull" \
  --lora_scale 1.0 \
  --steps 20
```

#### Option C: Google Colab Notebook
Create a new Colab cell:

```python
# Install dependencies
!pip install requests pyngrok

# Import the inference script
import requests

PUBLIC_URL = "https://xxxx-xxxx-xxxx.ngrok.io"  # Paste your URL here
PROMPT = "a transparent glass apple"

# Generate image
response = requests.post(
    f"{PUBLIC_URL}/api/generate",
    json={
        "prompt": PROMPT,
        "steps": 20,
        "lora_scale": 1.0,
    },
    timeout=120
)

if response.ok:
    print("✅ Generation started!")
    result = response.json()
    print(f"Image saved to: {result.get('output_path')}")
else:
    print(f"❌ Error: {response.text}")
```

---

## How It Works

```
Your Local Machine (GPU)          Internet           Google Colab
┌─────────────────────────┐                     ┌──────────────┐
│ ai-toolkit UI           │◄────ngrok tunnel───►│ Browser or   │
│ (port 8675)             │                     │ Python script│
│                         │                     │              │
│ ✅ Full VRAM access     │                     │ ✅ Anywhere  │
│ ✅ Full GPU access      │                     │ ✅ Any device│
└─────────────────────────┘                     └──────────────┘
```

**ngrok** creates a secure tunnel that forwards requests from a public URL to your local machine's port 8675.

---

## Advanced: Manual Tunneling

If you prefer not to use ngrok, you can use:

- **cloudflare tunnel**: `pip install cloudflare`
- **localtunnel**: `npm install -g localtunnel`
- **ssh reverse tunnel**: `ssh -R 80:localhost:8675 ssh.localhost.run`

But **ngrok is recommended** for simplicity and reliability.

---

## Troubleshooting

### "Failed to connect"
- Make sure `python run_with_public_url.py` is still running on your local machine
- Check that the URL is correct (copy-paste from the ngrok output)
- Check firewall/antivirus blocking port 8675

### "UI loads but can't submit jobs"
- The URL needs to include `http://` or `https://` prefix
- Make sure you're using the public ngrok URL, not `localhost`

### "ngrok tunnel keeps disconnecting"
- Free ngrok accounts have rate limits and inactivity timeouts
- Keep the terminal open while using it
- For long-term use, consider a paid ngrok plan

### Performance is slow
- ngrok adds slight latency, but should be fine for training monitoring
- For heavy API usage, consider setting up port forwarding instead

---

## Security Notes

⚠️ **Important**: The public URL exposes your local machine to the internet.

1. **Anyone with the URL can access your UI** — don't share it publicly
2. **The URL changes each time you restart** — no persistent public address
3. **Free ngrok sessions expire** — reconnect after ~8 hours of inactivity
4. **Consider authentication** if exposing to untrusted networks

For production/long-term use, set up:
- Static ngrok domains (paid)
- SSH key authentication
- VPN tunnel instead of ngrok

---

## What's Running

When you start `run_with_public_url.py`:

1. **Next.js Web UI** (port 8675) — React frontend with Monaco editor
2. **Node.js worker** — background job processor
3. **ngrok tunnel** — exposes port 8675 to public internet

The infrastructure stays on your local machine. Only the network tunnel is public.

---

## Using Colab to Control Your Local GPU

You can now use Google Colab's free resources (CPU, storage, internet) to:
- Control training from anywhere
- Monitor training in real-time
- Generate samples while away from home
- Organize datasets

But the **actual GPU/VRAM** is still on your local machine!

```python
# Colab cell
!pip install requests

# Control your local GPU
response = requests.get("https://xxxx.ngrok.io/api/system/info")
print(response.json())  # See local GPU status

# Submit training job
job_config = {...}
requests.post("https://xxxx.ngrok.io/api/jobs", json=job_config)
```

---

## Next Steps

- ✅ Start the UI with `python run_with_public_url.py`
- ✅ Paste the URL into your browser
- ✅ Use it from Colab or your phone!
- ✅ Run `colab_inference.py` for scripted generation
