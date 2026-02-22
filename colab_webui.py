"""
Google Colab notebook for running ai-toolkit web UI on Colab GPU.

This script:
1. Sets up the environment
2. Installs dependencies
3. Runs the web UI server
4. Exposes it publicly via ngrok
5. Prints the public URL

To use:
1. Open Google Colab: https://colab.research.google.com
2. Create a new notebook
3. Paste the code below into the first cell and run it

Or save this as a .ipynb file and upload to Colab.
"""

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================

!pip install -q pyngrok

# ============================================================================
# CELL 2: Clone ai-toolkit repo (if not already there)
# ============================================================================

import os
import subprocess

repo_path = "/content/ai-toolkit"

if not os.path.exists(repo_path):
    print("📥 Cloning ai-toolkit repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/ostris/ai-toolkit.git", repo_path],
        check=True
    )
else:
    print("✅ ai-toolkit already present")

os.chdir(repo_path)
print(f"📂 Working directory: {os.getcwd()}")

# ============================================================================
# CELL 3: Install ai-toolkit dependencies
# ============================================================================

print("📦 Installing ai-toolkit dependencies...")
subprocess.run(["pip", "install", "-q", "-e", "."], check=False)

# ============================================================================
# CELL 4: Set up Next.js UI
# ============================================================================

print("⚙️  Setting up Next.js UI...")
ui_path = "/content/ai-toolkit/ui"

# Install Node.js if needed
print("✅ Installing Node.js dependencies...")
subprocess.run(
    ["npm", "install", "--prefix", ui_path],
    capture_output=True,
    check=False
)

# Build Next.js
print("🔨 Building Next.js app...")
subprocess.run(
    ["npm", "run", "build", "--prefix", ui_path],
    capture_output=True,
    check=False
)

# ============================================================================
# CELL 5: Start the web UI with public tunnel
# ============================================================================

import time
from pyngrok import ngrok
from threading import Thread

UI_PORT = 8675

def start_ui():
    """Start the Next.js UI server"""
    subprocess.run(
        ["npm", "run", "start", "--prefix", ui_path],
        env={**os.environ, "PORT": str(UI_PORT)}
    )

def main():
    print("🚀 Starting ai-toolkit web UI on Colab...\n")

    # Start UI in background thread
    ui_thread = Thread(target=start_ui, daemon=True)
    ui_thread.start()

    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(10)

    # Create ngrok tunnel
    print("🌐 Creating public tunnel with ngrok...\n")
    try:
        public_url = ngrok.connect(UI_PORT, "http")
        print("✅ Public tunnel created!\n")
        print("=" * 70)
        print("🔗 PUBLIC URL (access the web UI here):")
        print(f"   {public_url}")
        print("=" * 70)
        print(f"\n📝 Features available:")
        print(f"   ✅ Create training jobs")
        print(f"   ✅ Generate images with LoRA")
        print(f"   ✅ Monitor training")
        print(f"   ✅ View system info (GPU/VRAM)")
        print(f"\n⏱️  Server running on Colab GPU!")
        print(f"   Keep this cell running to keep the server alive")
        print(f"   You can open new cells to run commands while this runs\n")

        # Keep tunnel alive
        ui_thread.join()

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ngrok.kill()

if __name__ == "__main__":
    main()

# ============================================================================
# Run it!
# ============================================================================

main()
