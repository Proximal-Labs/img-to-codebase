"""Shared configuration for screenshot-to-HTML RL training."""

import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-27B")
LORA_RANK = int(os.environ.get("LORA_RANK", 32))
RENDERER_NAME = "qwen3_5_disable_thinking"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", 8))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", 0))  # 0 = all
LR = float(os.environ.get("LR", 1e-5))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 4096))
KL_BETA = float(os.environ.get("KL_BETA", 0.02))
PPO_CLIP_LOW = float(os.environ.get("PPO_CLIP_LOW", 0.8))
PPO_CLIP_HIGH = float(os.environ.get("PPO_CLIP_HIGH", 1.2))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 15))  # 0 = disabled

# ── Reward ────────────────────────────────────────────────────────────────────
IMG_SIZE = int(os.environ.get("IMG_SIZE", 256))
VIEWPORT_W = int(os.environ.get("VIEWPORT_W", 1024))
VIEWPORT_H = int(os.environ.get("VIEWPORT_H", 768))

# ── Dataset ───────────────────────────────────────────────────────────────────
WEBSIGHT_TARGET = int(os.environ.get("WEBSIGHT_TARGET", 2000))
WEBSIGHT_CANDIDATES = int(os.environ.get("WEBSIGHT_CANDIDATES", 10000))
MAX_HTML_CHARS = int(os.environ.get("MAX_HTML_CHARS", 2000))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")
DESIGN2CODE_DIR = os.path.join(DATA_DIR, "design2code")
DESIGN2CODE_MANIFEST = os.path.join(DESIGN2CODE_DIR, "manifest.json")
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(PROJECT_DIR, "runs"))
MODEL_PATH_FILE = os.path.join(LOG_DIR, "model_path.txt")
EVAL_DIR = os.path.join(PROJECT_DIR, "eval_output")

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "Given a screenshot, output the HTML/CSS code that reproduces the visual appearance. "
    "You may use Tailwind CSS, inline styles, or a <style> block — whatever best matches the design. "
    "Wrap your code in ```html ... ```."
)
