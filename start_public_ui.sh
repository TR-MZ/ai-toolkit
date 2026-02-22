#!/bin/bash
# Start ai-toolkit web UI with public ngrok tunnel

set -e

echo "🚀 Starting ai-toolkit web UI with public tunnel..."
echo ""

# Check if pyngrok is installed
python3 -c "import pyngrok" 2>/dev/null || {
    echo "❌ pyngrok not installed!"
    echo ""
    echo "Install with:"
    echo "  pip install pyngrok"
    echo ""
    exit 1
}

# Run the Python script
python3 run_with_public_url.py
