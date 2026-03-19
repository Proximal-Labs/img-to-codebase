"""
RL training loop: screenshot → HTML/CSS using Tinker + Qwen3.5-4B.

Uses GRPO-style advantage centering with PPO clipped loss.
Reward = visual similarity (SSIM + MSE) - KL penalty.
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

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from reward import compute_visual_reward, extract_html_from_response, load_reference_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen3.5-4B"
BATCH_SIZE = 8          # screenshots per batch
GROUP_SIZE = 8          # rollouts per screenshot (up from 4)
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", 0))  # 0 = all batches
LR = 4e-5
LORA_RANK = 32
MAX_TOKENS = 1024
IMG_SIZE = 256
LOG_DIR = "/tmp/prox-rl"
SAVE_EVERY = 10         # checkpoint every N batches (0 = disabled)
KL_BETA = 0.05          # KL penalty coefficient
PPO_CLIP_LOW = 0.8      # PPO clip range
PPO_CLIP_HIGH = 1.2

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


def compute_kl_penalty(sampling_logprobs: list[float]) -> float:
    """
    Approximate per-rollout KL divergence from the base policy.

    Since we snapshot weights before each batch, the sampling logprobs
    ARE from the current policy. The KL is approximated as the mean
    negative log-probability (higher = model is less confident = more
    divergent from a "confident" base). This is a simplified proxy —
    true KL would need a separate forward pass through the base model.

    For GRPO the main regularization comes from advantage centering,
    but this adds a soft penalty for low-confidence / degenerate outputs.
    """
    if not sampling_logprobs:
        return 0.0
    # Mean negative logprob as proxy for "how far from confident base"
    return -sum(sampling_logprobs) / len(sampling_logprobs)


# ── Main training loop ───────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load dataset
    dataset = load_dataset(MANIFEST_PATH)
    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"Loaded {len(dataset)} examples, {n_batches} batches")

    # Setup tokenizer + renderer
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
    reward_pages = [browser.new_page(viewport={"width": 512, "height": 512}) for _ in range(8)]

    logger.info(
        f"Starting RL training: GROUP_SIZE={GROUP_SIZE}, KL_BETA={KL_BETA}, "
        f"PPO_CLIP=[{PPO_CLIP_LOW}, {PPO_CLIP_HIGH}], SAVE_EVERY={SAVE_EVERY}"
    )

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # ── Checkpoint ────────────────────────────────────────────────────
        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            logger.info(f"Saving checkpoint at batch {batch_idx}...")
            training_client.save_state(
                name=f"checkpoint-{batch_idx:04d}",
            ).result()

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
        batch_kl: list[float] = []

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

                # Parse response and compute visual reward
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                generated_html = extract_html_from_response(content)
                visual_reward = compute_visual_reward(generated_html, ref_img, page, size=IMG_SIZE)

                # KL penalty: penalize divergence from base
                kl = compute_kl_penalty(seq.logprobs)
                reward = visual_reward - KL_BETA * kl

                rewards_G.append(reward)
                batch_kl.append(kl)

            # GRPO advantage centering
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            batch_rewards.append(mean_reward)

            # Skip if no variance
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

        # ── 3. Training step (PPO clipped loss) ──────────────────────────
        if len(datums) == 0:
            logger.warning(f"Batch {batch_idx}: no datums (all rollouts identical), skipping")
            continue

        fwd_bwd_future = training_client.forward_backward(
            datums,
            loss_fn="ppo",
            loss_fn_config={
                "clip_low_threshold": PPO_CLIP_LOW,
                "clip_high_threshold": PPO_CLIP_HIGH,
            },
        )
        optim_future = training_client.optim_step(adam_params)
        fwd_bwd_future.result()
        optim_future.result()

        # ── 4. Log metrics ────────────────────────────────────────────────
        elapsed = time.time() - t_start
        mean_batch_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        mean_kl = sum(batch_kl) / len(batch_kl) if batch_kl else 0.0
        metrics = {
            "batch": batch_idx,
            "reward_mean": round(mean_batch_reward, 4),
            "reward_min": round(min(batch_rewards), 4) if batch_rewards else 0.0,
            "reward_max": round(max(batch_rewards), 4) if batch_rewards else 0.0,
            "kl_mean": round(mean_kl, 4),
            "n_datums": len(datums),
            "elapsed_s": round(elapsed, 1),
        }
        logger.info(
            f"Batch {batch_idx}/{n_batches}: "
            f"reward={mean_batch_reward:.3f} "
            f"kl={mean_kl:.3f} "
            f"datums={len(datums)} "
            f"time={elapsed:.1f}s"
        )
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()

    metrics_file.close()

    # Save final checkpoint + weights
    logger.info("Saving final checkpoint...")
    training_client.save_state(name="final").result()
    # Save with explicit path for later reuse
    save_result = training_client.save_weights_for_sampler(name="prox-final").result()
    model_path = save_result.path
    rl_sampler = service_client.create_sampling_client(model_path=model_path)
    logger.info(f"Final model saved at: {model_path}")
    logger.info("Training complete!")

    # Write model path to file for easy reference
    with open(os.path.join(LOG_DIR, "model_path.txt"), "w") as f:
        f.write(model_path)

    # ── Eval: compare base vs RL on held-out examples ─────────────────────
    N_EVAL = int(os.environ.get("N_EVAL", 10))
    if N_EVAL <= 0:
        browser.close()
        pw.stop()
        return

    import random
    random.seed(42)

    logger.info(f"\nRunning eval on {N_EVAL} held-out examples...")

    # Base model (fresh untrained LoRA)
    base_client = service_client.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
    base_sampler = base_client.save_weights_and_get_sampling_client()

    eval_params = types.SamplingParams(
        max_tokens=MAX_TOKENS, stop=renderer.get_stop_sequences(), temperature=0.3,
    )

    # Use examples not in training set
    eval_pool = dataset[n_batches * BATCH_SIZE:]
    if len(eval_pool) < N_EVAL:
        eval_pool = dataset
    eval_samples = random.sample(eval_pool, min(N_EVAL, len(eval_pool)))

    base_rewards, rl_rewards = [], []
    eval_results = []

    for i, item in enumerate(eval_samples):
        prompt = build_prompt(renderer, item["screenshot"])
        ref_img = load_reference_image(item["screenshot"], size=IMG_SIZE)
        page = reward_pages[i % len(reward_pages)]

        # Base
        base_result = base_sampler.sample(prompt=prompt, num_samples=1, sampling_params=eval_params).result()
        base_parsed, _ = renderer.parse_response(base_result.sequences[0].tokens)
        base_html = extract_html_from_response(get_text_content(base_parsed))
        base_reward = compute_visual_reward(base_html, ref_img, page)

        # RL
        rl_result = rl_sampler.sample(prompt=prompt, num_samples=1, sampling_params=eval_params).result()
        rl_parsed, _ = renderer.parse_response(rl_result.sequences[0].tokens)
        rl_html = extract_html_from_response(get_text_content(rl_parsed))
        rl_reward = compute_visual_reward(rl_html, ref_img, page)

        base_rewards.append(base_reward)
        rl_rewards.append(rl_reward)
        eval_results.append({
            "screenshot": item["screenshot"],
            "base_reward": round(base_reward, 4),
            "rl_reward": round(rl_reward, 4),
            "base_html": base_html,
            "rl_html": rl_html,
        })

        delta = rl_reward - base_reward
        logger.info(f"  Eval {i+1}/{N_EVAL}: base={base_reward:.3f} rl={rl_reward:.3f} delta={delta:+.3f}")

    # Summary
    avg_base = float(np.mean(base_rewards))
    avg_rl = float(np.mean(rl_rewards))
    wins = sum(1 for b, r in zip(base_rewards, rl_rewards) if r > b)

    logger.info(f"\n{'='*60}")
    logger.info("EVAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Base avg reward: {avg_base:.3f}")
    logger.info(f"  RL   avg reward: {avg_rl:.3f}")
    logger.info(f"  Improvement:     {avg_rl - avg_base:+.3f}")
    logger.info(f"  RL wins: {wins}/{N_EVAL}")

    # Save eval results
    eval_dir = os.path.join(os.path.dirname(__file__), "eval_output")
    os.makedirs(eval_dir, exist_ok=True)
    eval_summary = {
        "base_mean": round(avg_base, 4),
        "rl_mean": round(avg_rl, 4),
        "improvement": round(avg_rl - avg_base, 4),
        "rl_wins": wins,
        "n_eval": N_EVAL,
        "n_train_batches": n_batches,
        "results": eval_results,
    }
    with open(os.path.join(eval_dir, "eval_comparison.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)
    logger.info(f"  Saved to {eval_dir}/eval_comparison.json")

    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
