"""
RL training loop: screenshot → HTML/CSS using Tinker + Qwen3.5-4B.

Uses GRPO-style advantage centering with importance sampling loss.
Reward = SSIM between original screenshot and rendered generated HTML.
"""

import json
import logging
import os
import time

import numpy as np
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from PIL import Image
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import ImagePart, TextPart, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from reward import compute_visual_reward, extract_html_from_response, load_reference_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3.5-4B"
BATCH_SIZE = 8          # screenshots per batch
GROUP_SIZE = 4          # rollouts per screenshot
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", 0))  # 0 = all batches
LR = 4e-5
LORA_RANK = 32
MAX_TOKENS = 1024       # enough for web-sourced HTML snippets
IMG_SIZE = 256          # resize for SSIM comparison
LOG_DIR = "/tmp/prox-rl"

SYSTEM_PROMPT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "Given a screenshot, output ONLY the HTML/CSS code that reproduces the visual appearance. "
    "Use inline styles or a <style> block. Do not include <html>, <head>, or <body> wrapper tags. "
    "Wrap your code in ```html ... ```."
)

MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "data", "manifest.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        return json.load(f)


def build_prompt(renderer, screenshot_path: str):
    """Build a multimodal prompt with the screenshot image + instruction."""
    img = Image.open(screenshot_path).convert("RGB")

    convo = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Generate the HTML/CSS snippet that reproduces this screenshot."},
            ],
        },
    ]
    return renderer.build_generation_prompt(convo)


# ── Main training loop ───────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load dataset
    dataset = load_dataset(MANIFEST_PATH)
    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"Loaded {len(dataset)} examples, {n_batches} batches")

    # Setup tokenizer + renderer (VLM needs image_processor)
    # Use disable_thinking renderer so the model outputs HTML directly
    # instead of spending tokens on chain-of-thought reasoning
    tokenizer = get_tokenizer(MODEL)
    renderer_name = "qwen3_5_disable_thinking"

    from transformers import AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor=image_processor)
    logger.info(f"Using renderer: {renderer_name}")

    # Setup Tinker clients
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=MODEL, rank=LORA_RANK,
    )

    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        stop=renderer.get_stop_sequences(),
        temperature=0.7,
    )
    adam_params = types.AdamParams(
        learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8,
    )

    # Metrics log
    metrics_path = os.path.join(LOG_DIR, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    # Playwright for reward computation
    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    # Pool of pages for parallel reward computation
    reward_pages = [browser.new_page(viewport={"width": 512, "height": 512}) for _ in range(4)]

    logger.info("Starting RL training loop")

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # ── 1. Sample rollouts ────────────────────────────────────────────
        sampling_client = training_client.save_weights_and_get_sampling_client()

        futures = []
        prompts = []
        ref_images = []

        for item in batch:
            prompt = build_prompt(renderer, item["screenshot"])
            future = sampling_client.sample(
                prompt=prompt,
                num_samples=GROUP_SIZE,
                sampling_params=sampling_params,
            )
            futures.append(future)
            prompts.append(prompt)
            ref_images.append(load_reference_image(item["screenshot"], size=IMG_SIZE))

        # ── 2. Compute rewards & build datums ─────────────────────────────
        datums: list[types.Datum] = []
        batch_rewards: list[float] = []

        for idx, (future, prompt, ref_img) in enumerate(
            tqdm(
                zip(futures, prompts, ref_images),
                total=len(futures),
                desc=f"Batch {batch_idx}",
            )
        ):
            result = future.result()
            rewards_G: list[float] = []
            tokens_G: list[list[int]] = []
            logprobs_G: list[list[float]] = []

            page = reward_pages[idx % len(reward_pages)]

            for seq in result.sequences:
                tokens_G.append(seq.tokens)
                logprobs_G.append(seq.logprobs)

                # Parse response and compute reward
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                generated_html = extract_html_from_response(content)
                reward = compute_visual_reward(generated_html, ref_img, page, size=IMG_SIZE)
                rewards_G.append(reward)

            # GRPO advantage centering
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            batch_rewards.append(mean_reward)

            # Skip if no variance (all same reward)
            if all(a == 0.0 for a in advantages_G):
                continue

            # Build training datums
            for tokens, logprobs, advantage in zip(tokens_G, logprobs_G, advantages_G):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                assert (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                )

                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums.append(datum)

        # ── 3. Training step ──────────────────────────────────────────────
        if len(datums) == 0:
            logger.warning(f"Batch {batch_idx}: no datums (all rollouts identical), skipping")
            continue

        fwd_bwd_future = training_client.forward_backward(datums, loss_fn="importance_sampling")
        optim_future = training_client.optim_step(adam_params)
        fwd_bwd_future.result()
        optim_future.result()

        # ── 4. Log metrics ────────────────────────────────────────────────
        elapsed = time.time() - t_start
        mean_batch_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        metrics = {
            "batch": batch_idx,
            "reward_mean": mean_batch_reward,
            "reward_min": min(batch_rewards) if batch_rewards else 0.0,
            "reward_max": max(batch_rewards) if batch_rewards else 0.0,
            "n_datums": len(datums),
            "elapsed_s": round(elapsed, 1),
        }
        logger.info(
            f"Batch {batch_idx}/{n_batches}: "
            f"reward={mean_batch_reward:.3f} "
            f"datums={len(datums)} "
            f"time={elapsed:.1f}s"
        )
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()

    # ── Cleanup ───────────────────────────────────────────────────────────
    browser.close()
    pw.stop()
    metrics_file.close()

    # Save final weights
    logger.info("Saving final weights...")
    final_client = training_client.save_weights_and_get_sampling_client(name="prox-final")
    logger.info("Final model saved")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
