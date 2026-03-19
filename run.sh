#!/bin/bash
set -e

# Ensure API key is set
if [ -z "$TINKER_API_KEY" ]; then
    if [ -f .env ]; then
        source .env
        export TINKER_API_KEY
    else
        echo "Error: TINKER_API_KEY not set. Export it or create a .env file (see .env.example)."
        exit 1
    fi
fi

echo "=== Step 1: Install dependencies ==="
pip install -r requirements.txt
pip install tinker-cookbook
playwright install chromium

echo ""
echo "=== Step 2: Generate dataset ==="
python generate_dataset_web.py

echo ""
echo "=== Step 3: Train + Eval ==="
python train.py
