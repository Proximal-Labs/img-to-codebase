# prox — Screenshot-to-HTML RL Training

Train a vision-language model (Qwen3.5-27B) to convert webpage screenshots into HTML/CSS using reinforcement learning with [Tinker](https://docs.tinker.dev/).

## Overview

The pipeline:
1. **Dataset**: Downloads and filters HTML/screenshot pairs from [WebSight v0.2](https://huggingface.co/datasets/HuggingFaceM4/WebSight)
2. **Training**: GRPO-style RL with PPO clipped loss — the model generates HTML from a screenshot and gets rewarded based on how well its output matches the reference
3. **Eval**: Compares base vs RL-trained model on held-out examples, saving side-by-side screenshots

### Reward Signal

Multi-signal reward combining DOM-level and image-level metrics:
- **DOM**: block position (IoU), text content, background/text color, font family/size
- **Image**: CLIP perceptual similarity, SSIM + MSE pixel similarity

## Setup

```bash
# Clone and enter the repo
git clone <repo-url> && cd prox

# Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install tinker-cookbook
playwright install chromium
```

Set your Tinker API key:

```bash
export TINKER_API_KEY=your-key-here
# Or create a .env file (see .env.example)
```

## Usage

### Quick start (full pipeline)

```bash
./run.sh
```

### Step by step

```bash
# 1. Generate dataset (downloads WebSight, filters, renders screenshots)
python generate_dataset_web.py

# 2. Train (RL loop with Tinker)
python train.py

# 3. Eval (compare base vs RL model)
python eval.py
```

### Alternative dataset generators

```bash
# Synthetic HTML snippets (no download needed)
python generate_dataset.py

# Design2Code benchmark
python download_design2code.py
```

### Eval options

```bash
python eval.py --n 20                          # more eval examples
python eval.py --model_path "tinker://..."     # specific model checkpoint
python eval.py --base_only                     # base model only
```

## Configuration

All config lives in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-27B` | Base model |
| `LORA_RANK` | `32` | LoRA rank |
| `BATCH_SIZE` | `8` | Batch size |
| `GROUP_SIZE` | `8` | Rollouts per example (GRPO) |
| `MAX_TOKENS` | `1024` | Max generation tokens |
| `KL_BETA` | `0.05` | KL penalty coefficient |
| `SAVE_EVERY` | `15` | Checkpoint frequency (batches) |
| `LR` | `4e-5` | Learning rate |
| `WEBSIGHT_TARGET` | `2000` | Dataset size |
| `MAX_HTML_CHARS` | `2000` | Max HTML length filter |
| `LOG_DIR` | `./runs` | Training logs and checkpoints |

## Project Structure

```
prox/
  config.py                 # Shared configuration
  generate_dataset_web.py   # WebSight dataset generator
  generate_dataset.py       # Synthetic dataset generator
  download_design2code.py   # Design2Code benchmark downloader
  train.py                  # RL training loop
  eval.py                   # Evaluation script
  reward.py                 # Multi-signal reward function
  run.sh                    # End-to-end pipeline script
  requirements.txt          # Python dependencies
  data/                     # Generated datasets (gitignored)
  runs/                     # Training logs and checkpoints (gitignored)
  eval_output/              # Evaluation outputs (gitignored)
```
