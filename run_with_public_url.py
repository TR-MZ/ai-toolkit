#!/usr/bin/env python3
"""
Start the ai-toolkit web UI with a public localtunnel.
This allows you to access the web UI from anywhere, including Google Colab.

Requirements:
  npm install -g localtunnel

Usage:
  python run_with_public_url.py

The script will print a public URL like: https://xxxx-xxxx-xxxx.loca.lt
Use that URL to access the UI from Colab or anywhere else.
"""

import subprocess
import time
import sys
import os

def check_localtunnel():
    """Check if localtunnel is installed"""
    result = subprocess.run(["which", "lt"], capture_output=True)
    if result.returncode != 0:
        print("❌ localtunnel not installed!")
        print("\nInstall it with:")
        print("  npm install -g localtunnel")
        sys.exit(1)

# Port where Next.js UI runs
UI_PORT = 8675

def main():
    check_localtunnel()

    print(f"🚀 Starting ai-toolkit UI on port {UI_PORT}...")
    print(f"🌐 Setting up public tunnel with localtunnel...\n")

    # Start the UI in the background
    ui_process = subprocess.Popen(
        ["npm", "run", "start"],
        cwd="/home/user/Documents/ai-toolkit/ui",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give the UI a moment to start
    time.sleep(5)

    # Start localtunnel tunnel
    try:
        print(f"Creating tunnel to localhost:{UI_PORT}...\n")
        tunnel_process = subprocess.Popen(
            ["lt", "--port", str(UI_PORT), "--local-host", "127.0.0.1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        # Read the tunnel URL from output
        url_found = False
        for line in tunnel_process.stdout:
            print(line.strip())
            if "your url is:" in line.lower() or ".loca.lt" in line:
                if not url_found:
                    print(f"\n{'='*60}")
                    print(f"✅ Public tunnel created!")
                    print(f"{'='*60}\n")
                    print(f"Local:  http://localhost:{UI_PORT}")
                    print(f"Public: {line.strip()}\n")
                    print("Press Ctrl+C to stop the server and tunnel\n")
                    url_found = True

        # Keep processes alive
        tunnel_process.wait()

    except KeyboardInterrupt:
        print("\n⏹️  Stopping server...")
        tunnel_process.terminate()
        ui_process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")
        ui_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
