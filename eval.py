"""
Eval: compare base Qwen3.5-4B vs RL-trained model on held-out screenshots.

Usage:
    # Compare base vs RL (loads model path from /tmp/prox-rl/model_path.txt)
    python eval.py

    # Specify RL model path explicitly
    python eval.py --model_path "tinker://abc123/weights/prox-final"

    # Base-only eval
    python eval.py --base_only

    # More examples
    python eval.py --n 20
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import tinker
from tinker import types
from PIL import Image
from playwright.sync_api import sync_playwright

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoImageProcessor
from reward import (
    extract_html_from_response,
    load_reference_image,
    compute_visual_reward,
)

MODEL = "Qwen/Qwen3.5-4B"
LORA_RANK = 32
MAX_TOKENS = 1024
IMG_SIZE = 256
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "data", "manifest.json")
MODEL_PATH_FILE = "/tmp/prox-rl/model_path.txt"

SYSTEM_PROMPT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "Given a screenshot, output ONLY the HTML/CSS code that reproduces the visual appearance. "
    "Use inline styles or a <style> block. Do not include <html>, <head>, or <body> wrapper tags. "
    "Wrap your code in ```html ... ```."
)


def log(msg):
    print(msg, flush=True)


def build_prompt(renderer, screenshot_path: str):
    img = Image.open(screenshot_path).convert("RGB")
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Generate the HTML/CSS snippet that reproduces this screenshot."},
            ],
        },
    ]
    return renderer.build_generation_prompt(convo)


def eval_model(tag, sampler, renderer, samples, eval_params, reward_pages):
    """Evaluate a model on samples, return rewards and results."""
    rewards = []
    results = []

    for i, item in enumerate(samples):
        prompt = build_prompt(renderer, item["screenshot"])
        ref_img = load_reference_image(item["screenshot"], size=IMG_SIZE)
        page = reward_pages[i % len(reward_pages)]

        result = sampler.sample(prompt=prompt, num_samples=1, sampling_params=eval_params).result()
        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        content = get_text_content(parsed_msg)
        generated_html = extract_html_from_response(content)
        reward = compute_visual_reward(generated_html, ref_img, page)

        rewards.append(reward)
        results.append({
            "screenshot": item["screenshot"],
            "generated_html": generated_html,
            "reward": round(reward, 4),
        })

        log(f"  [{tag}] {i+1}/{len(samples)}: reward={reward:.3f}  ({os.path.basename(item['screenshot'])})")

    return rewards, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of eval examples")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Tinker model path for RL weights (auto-detected from last training run)")
    parser.add_argument("--base_only", action="store_true", help="Only eval base model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Find RL model path
    rl_model_path = args.model_path
    if not rl_model_path and not args.base_only:
        if os.path.exists(MODEL_PATH_FILE):
            with open(MODEL_PATH_FILE) as f:
                rl_model_path = f.read().strip()
            log(f"Loaded RL model path: {rl_model_path}")
        else:
            log(f"No model path found at {MODEL_PATH_FILE}. Run train.py first, or pass --model_path.")
            log("Running base-only eval.")
            args.base_only = True

    with open(MANIFEST_PATH) as f:
        dataset = json.load(f)

    samples = random.sample(dataset, min(args.n, len(dataset)))

    # Setup
    tokenizer = get_tokenizer(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer("qwen3_5_disable_thinking", tokenizer, image_processor=image_processor)

    eval_params = types.SamplingParams(
        max_tokens=MAX_TOKENS, stop=renderer.get_stop_sequences(), temperature=0.3,
    )

    service_client = tinker.ServiceClient()

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    reward_pages = [browser.new_page(viewport={"width": 512, "height": 512}) for _ in range(8)]

    out_dir = os.path.join(os.path.dirname(__file__), "eval_output")
    os.makedirs(out_dir, exist_ok=True)

    # Base model
    log("\nEvaluating base model (untrained LoRA)...")
    base_client = service_client.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
    base_sampler = base_client.save_weights_and_get_sampling_client()
    base_rewards, base_results = eval_model("base", base_sampler, renderer, samples, eval_params, reward_pages)

    # RL model
    rl_rewards, rl_results = None, None
    if not args.base_only:
        log(f"\nEvaluating RL model ({rl_model_path})...")
        rl_sampler = service_client.create_sampling_client(model_path=rl_model_path)
        rl_rewards, rl_results = eval_model("rl", rl_sampler, renderer, samples, eval_params, reward_pages)

    # Summary
    log(f"\n{'='*60}")
    log("EVAL RESULTS")
    log(f"{'='*60}")

    avg_base = float(np.mean(base_rewards))
    log(f"  Base model:  mean={avg_base:.3f}  std={np.std(base_rewards):.3f}  "
        f"min={np.min(base_rewards):.3f}  max={np.max(base_rewards):.3f}")

    if rl_rewards is not None:
        avg_rl = float(np.mean(rl_rewards))
        wins = sum(1 for b, r in zip(base_rewards, rl_rewards) if r > b)
        ties = sum(1 for b, r in zip(base_rewards, rl_rewards) if r == b)

        log(f"  RL model:    mean={avg_rl:.3f}  std={np.std(rl_rewards):.3f}  "
            f"min={np.min(rl_rewards):.3f}  max={np.max(rl_rewards):.3f}")
        log(f"  Improvement: {avg_rl - avg_base:+.3f}")
        log(f"  RL wins: {wins}/{args.n}  ties: {ties}/{args.n}")

        log(f"\n  Per-example:")
        for i, (br, rr) in enumerate(zip(base_rewards, rl_rewards)):
            marker = ">" if rr > br else ("<" if rr < br else "=")
            log(f"    {i+1}: base={br:.3f}  {marker}  rl={rr:.3f}  (delta={rr-br:+.3f})")

        # Save comparison
        comparison = {
            "base_mean": round(avg_base, 4),
            "rl_mean": round(avg_rl, 4),
            "improvement": round(avg_rl - avg_base, 4),
            "rl_wins": wins,
            "n_eval": args.n,
            "rl_model_path": rl_model_path,
            "per_example": [
                {"base": round(b, 4), "rl": round(r, 4), "delta": round(r - b, 4)}
                for b, r in zip(base_rewards, rl_rewards)
            ],
        }
        with open(os.path.join(out_dir, "eval_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

    log(f"\nResults saved to {out_dir}/")

    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
