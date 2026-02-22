# ⚡ Quick Start: Access from Colab in 2 Steps

## Step 1: Start the public tunnel on your local machine

```bash
pip install pyngrok
python run_with_public_url.py
```

You'll see:
```
🔗 PUBLIC URL (use this to access from Colab):
   https://xxxx-xxxx-xxxx.ngrok.io
```

**Copy that URL** ↑

## Step 2: Use from Colab

### A. Web UI (easiest)
Paste the URL into your browser and use the web UI normally.

### B. Python script
```bash
python colab_inference.py --url "https://xxxx-xxxx-xxxx.ngrok.io" --prompt "a transparent glass skull"
```

### C. Colab notebook
```python
import requests

url = "https://xxxx-xxxx-xxxx.ngrok.io"  # Paste your URL

# Check connection
response = requests.get(f"{url}/api/system")
print("✅ Connected!" if response.ok else "❌ Connection failed")
```

---

## That's it! 🎉

Your local GPU is now accessible from anywhere!

For more details, see [COLAB_SETUP.md](COLAB_SETUP.md)
