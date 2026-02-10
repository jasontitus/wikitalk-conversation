#!/usr/bin/env bash
# WikiTalk Quick Start
# ====================
# One-liner to get up and running:
#   curl -sSL <this-repo>/quickstart.sh | bash
#   -- or --
#   ./quickstart.sh
#   ./quickstart.sh --topic "Roman Empire"
#
# What it does:
#   1. Creates a Python venv (if not already in one)
#   2. Installs requirements
#   3. Downloads ~100 Wikipedia articles via the API
#   4. Chunks & indexes them for semantic search
#   5. Launches WikiTalk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  WikiTalk Quick Start"
echo "============================================================"
echo

# -------------------------------------------------------------------
# 1. Python environment
# -------------------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -d ".venv" ]; then
        echo "Creating Python virtual environment (.venv)..."
        python3 -m venv .venv
    fi
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "Using active virtual environment: $VIRTUAL_ENV"
fi

echo "Python: $(python3 --version) at $(which python3)"
echo

# -------------------------------------------------------------------
# 2. Install dependencies
# -------------------------------------------------------------------
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  Done."
echo

# -------------------------------------------------------------------
# 3. Check for LLM server
# -------------------------------------------------------------------
LLM_URL="http://localhost:1234/v1/models"
if curl -s --connect-timeout 2 "$LLM_URL" > /dev/null 2>&1; then
    echo "LLM server detected at localhost:1234"
else
    echo "WARNING: No LLM server detected at localhost:1234"
    echo "  WikiTalk needs a local LLM server (LM Studio, llama.cpp, Ollama, etc.)"
    echo "  Start one before chatting. Recommended model: Qwen3-8B"
    echo
fi

# -------------------------------------------------------------------
# 4. Download, process, and launch
# -------------------------------------------------------------------
python3 quickstart.py "$@"
