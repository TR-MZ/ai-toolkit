#!/usr/bin/env python3
"""
Start the ai-toolkit web UI with a public ngrok tunnel.
This allows you to access the web UI from anywhere, including Google Colab.

Requirements:
  pip install pyngrok

Usage:
  python run_with_public_url.py

The script will print a public URL like: https://xxxx-xxx-xxx-xxx.ngrok.io
Use that URL to access the UI from Colab or anywhere else.
"""

import subprocess
import time
import sys
import os

try:
    from pyngrok import ngrok
except ImportError:
    print("❌ pyngrok not installed!")
    print("\nInstall it with:")
    print("  pip install pyngrok")
    sys.exit(1)

# Port where Next.js UI runs
UI_PORT = 8675

def main():
    print(f"🚀 Starting ai-toolkit UI on port {UI_PORT}...")
    print(f"🌐 Setting up public tunnel with ngrok...\n")

    # Start the UI in the background
    ui_process = subprocess.Popen(
        ["npm", "run", "start"],
        cwd="/home/user/Documents/ai-toolkit/ui",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give the UI a moment to start
    time.sleep(5)

    # Set up ngrok tunnel
    try:
        public_url = ngrok.connect(UI_PORT, "http")
        print("✅ Public tunnel created!")
        print(f"\n{'='*60}")
        print(f"🔗 PUBLIC URL (use this to access from Colab):")
        print(f"   {public_url}")
        print(f"{'='*60}\n")
        print(f"Local:  http://localhost:{UI_PORT}")
        print(f"Public: {public_url}\n")
        print("Press Ctrl+C to stop the server and tunnel\n")

        # Keep the process alive
        ui_process.wait()
    except Exception as e:
        print(f"❌ Error setting up ngrok: {e}")
        ui_process.terminate()
        sys.exit(1)
    finally:
        ngrok.kill()
        ui_process.terminate()

if __name__ == "__main__":
    main()
