"""
Simple RL training: screenshot → HTML, one shot, SSIM reward.

No analyze, no turns, no re-rendering reference HTML.
Just: model sees screenshot, generates HTML, we render it, compare SSIM
against the original screenshot.
"""

import json
import logging
import os
import random
import sys
import time
import io

import numpy as np
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from PIL import Image
from playwright.sync_api import sync_playwright
from tqdm import tqdm
from transformers import AutoImageProcessor
from skimage.metrics import structural_similarity as ssim_fn

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    MODEL, LORA_RANK, RENDERER_NAME, LR, KL_BETA,
    PPO_CLIP_LOW, PPO_CLIP_HIGH, LOG_DIR, MANIFEST_PATH,
)
from reward import render_html, extract_html_from_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", 2))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", 0))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 5))
TOKENS = int(os.environ.get("TOKENS", 2048))
VIEWPORT = {"width": 1280, "height": 720}

SYSTEM_PROMPT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "You may use Tailwind CSS, inline styles, or a <style> block. "
    "Wrap your code in ```html ... ```."
)


def take_screenshot(page) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def compute_ssim(ref: np.ndarray, gen: np.ndarray) -> float:
    if ref.shape != gen.shape:
        gen = np.array(Image.fromarray(gen).resize((ref.shape[1], ref.shape[0])))
    return float(ssim_fn(ref, gen, channel_axis=2, data_range=255))


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load dataset
    with open(MANIFEST_PATH) as f:
        dataset = json.load(f)
    random.shuffle(dataset)

    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"{len(dataset)} examples, {n_batches} batches (BS={BATCH_SIZE}, GS={GROUP_SIZE})")

    # Setup
    tokenizer = get_tokenizer(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer, image_processor=image_processor)

    service_client = tinker.ServiceClient()
    resume_path = os.environ.get("RESUME_FROM")
    if resume_path:
        logger.info(f"Resuming: {resume_path}")
        training_client = service_client.create_training_client_from_state_with_optimizer(resume_path)
    else:
        training_client = service_client.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)

    sampling_params = types.SamplingParams(
        max_tokens=TOKENS, stop=renderer.get_stop_sequences(), temperature=0.7,
    )
    adam_params = types.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)

    metrics_path = os.path.join(LOG_DIR, "metrics_simple.jsonl")
    metrics_file = open(metrics_path, "a")

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    page = browser.new_page(viewport=VIEWPORT)

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            logger.info(f"Checkpoint {batch_idx}...")
            training_client.save_state(name=f"checkpoint-{batch_idx:04d}").result()

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Load reference screenshots (original images, not re-rendered HTML)
        ref_images = []
        prompts = []
        futures = []

        for item in batch:
            ref_img = np.array(Image.open(item["screenshot"]).convert("RGB"))
            ref_images.append(ref_img)

            ref_pil = Image.open(item["screenshot"]).convert("RGB")
            prompt = renderer.build_generation_prompt([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": ref_pil},
                    {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
                ]},
            ])
            prompts.append(prompt)

            # Fire all sampling in parallel
            futures.append(sampling_client.sample(
                prompt=prompt, num_samples=GROUP_SIZE, sampling_params=sampling_params,
            ))

        # Collect results
        datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_kl: list[float] = []

        for idx in tqdm(range(len(batch)), desc=f"B{batch_idx}"):
            result = futures[idx].result()
            ref_img = ref_images[idx]
            prompt = prompts[idx]

            rewards_G = []
            tokens_G = []
            logprobs_G = []

            for seq in result.sequences:
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                html = extract_html_from_response(content)

                if html is None:
                    reward = -1.0
                else:
                    try:
                        render_html(page, html)
                        gen_img = take_screenshot(page)
                        reward = 2.0 * compute_ssim(ref_img, gen_img) - 1.0  # Scale to [-1, 1]
                    except Exception:
                        reward = -1.0

                kl = -sum(seq.logprobs) / len(seq.logprobs) if seq.logprobs else 0.0
                rewards_G.append(reward - KL_BETA * kl)
                tokens_G.append(list(seq.tokens))
                logprobs_G.append(list(seq.logprobs))
                batch_kl.append(kl)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            batch_rewards.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            for tokens, logprobs, advantage in zip(tokens_G, logprobs_G, advantages_G):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                assert model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages)

                datums.append(types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                ))

        if not datums:
            logger.warning(f"Batch {batch_idx}: no datums")
            continue

        fwd_bwd = training_client.forward_backward(
            datums, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW, "clip_high_threshold": PPO_CLIP_HIGH},
        )
        optim = training_client.optim_step(adam_params)
        fwd_bwd.result()
        optim.result()

        elapsed = time.time() - t_start
        mean_r = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        mean_kl = sum(batch_kl) / len(batch_kl) if batch_kl else 0.0
        logger.info(f"Batch {batch_idx}/{n_batches}: reward={mean_r:.3f} kl={mean_kl:.3f} datums={len(datums)} time={elapsed:.0f}s")
        metrics_file.write(json.dumps({"batch": batch_idx, "reward": round(mean_r, 4), "kl": round(mean_kl, 4), "datums": len(datums), "time": round(elapsed, 1)}) + "\n")
        metrics_file.flush()

    metrics_file.close()
    logger.info("Saving final...")
    training_client.save_state(name="final").result()
    result = training_client.save_weights_for_sampler(name="prox-simple-final").result()
    logger.info(f"Saved: {result.path}")
    with open(os.path.join(LOG_DIR, "model_path_simple.txt"), "w") as f:
        f.write(result.path)

    browser.close()
    pw.stop()
    logger.info("Done!")


if __name__ == "__main__":
    main()
