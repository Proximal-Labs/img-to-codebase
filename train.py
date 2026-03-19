"""
RL training loop: screenshot → HTML/CSS using Tinker.

Uses GRPO-style advantage centering with PPO clipped loss.
Reward = DOM block matching + text/color/font match + CLIP + visual similarity - KL penalty.
"""

import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from PIL import Image
from playwright.sync_api import sync_playwright
from tqdm import tqdm
from transformers import AutoImageProcessor

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    MODEL, LORA_RANK, RENDERER_NAME, BATCH_SIZE, GROUP_SIZE, MAX_BATCHES,
    LR, MAX_TOKENS, KL_BETA, PPO_CLIP_LOW, PPO_CLIP_HIGH, SAVE_EVERY,
    IMG_SIZE, VIEWPORT_W, VIEWPORT_H, MANIFEST_PATH, DESIGN2CODE_MANIFEST,
    LOG_DIR, SYSTEM_PROMPT, EVAL_DIR,
)
from reward import (
    compute_visual_reward, extract_html_from_response,
    load_reference_image, render_html_to_file,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        return json.load(f)


def load_training_data() -> list[dict]:
    """
    Load training data with curriculum ordering: easy (short HTML) → hard (long HTML).

    Merges WebSight + Design2Code, sorts by HTML length so the model
    learns simple layouts before tackling complex pages.
    """
    dataset = load_dataset(MANIFEST_PATH)
    logger.info(f"  WebSight: {len(dataset)} examples")

    if os.path.exists(DESIGN2CODE_MANIFEST):
        d2c = load_dataset(DESIGN2CODE_MANIFEST)
        # Filter to pages the model can actually generate (~8K chars ≈ 4K tokens)
        d2c = [x for x in d2c if len(x.get("html", "")) < 8000]
        logger.info(f"  Design2Code: {len(d2c)} examples (filtered to <8K chars)")
        dataset.extend(d2c)

    # Curriculum: sort by HTML length (easy → hard)
    dataset.sort(key=lambda x: len(x.get("html", "")))
    logger.info(f"  Curriculum: {len(dataset)} total, "
                f"shortest={len(dataset[0]['html'])} chars, "
                f"longest={len(dataset[-1]['html'])} chars")
    return dataset


def build_prompt(renderer, screenshot_path: str):
    """Build a multimodal prompt with the screenshot image + instruction."""
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


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    dataset = load_training_data()
    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"Loaded {len(dataset)} examples, {n_batches} batches")

    # Setup tokenizer + renderer
    tokenizer = get_tokenizer(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer, image_processor=image_processor)
    logger.info(f"Using renderer: {RENDERER_NAME}")

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

    metrics_path = os.path.join(LOG_DIR, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    # Playwright for reward computation
    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    reward_pages = [
        browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
        for _ in range(GROUP_SIZE)
    ]

    logger.info(
        f"Config: GROUP_SIZE={GROUP_SIZE}, KL_BETA={KL_BETA}, "
        f"PPO_CLIP=[{PPO_CLIP_LOW}, {PPO_CLIP_HIGH}], SAVE_EVERY={SAVE_EVERY}"
    )

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # Checkpoint
        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            logger.info(f"Saving checkpoint at batch {batch_idx}...")
            training_client.save_state(name=f"checkpoint-{batch_idx:04d}").result()

        # 1. Sample rollouts
        sampling_client = training_client.save_weights_and_get_sampling_client()
        futures, prompts, ref_images, ref_htmls = [], [], [], []

        for item in batch:
            prompt = build_prompt(renderer, item["screenshot"])
            future = sampling_client.sample(
                prompt=prompt, num_samples=GROUP_SIZE, sampling_params=sampling_params,
            )
            futures.append(future)
            prompts.append(prompt)
            ref_images.append(load_reference_image(item["screenshot"], size=IMG_SIZE))
            ref_htmls.append(item.get("reference_html") or item["html"])

        # 2. Compute rewards & build datums
        datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_kl: list[float] = []

        for idx, (future, prompt, ref_img, ref_html) in enumerate(
            tqdm(zip(futures, prompts, ref_images, ref_htmls), total=len(futures), desc=f"Batch {batch_idx}")
        ):
            result = future.result()
            rewards_G: list[float] = []
            tokens_G: list[list[int]] = []
            logprobs_G: list[list[float]] = []
            page = reward_pages[idx % len(reward_pages)]

            for seq in result.sequences:
                tokens_G.append(seq.tokens)
                logprobs_G.append(seq.logprobs)

                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                generated_html = extract_html_from_response(content)
                visual_reward = compute_visual_reward(
                    generated_html, ref_img, page, size=IMG_SIZE, reference_html=ref_html,
                )

                kl = -sum(seq.logprobs) / len(seq.logprobs) if seq.logprobs else 0.0
                rewards_G.append(visual_reward - KL_BETA * kl)
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

        # 3. Training step
        if len(datums) == 0:
            logger.warning(f"Batch {batch_idx}: no datums, skipping")
            continue

        training_client.forward_backward(
            datums, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW, "clip_high_threshold": PPO_CLIP_HIGH},
        ).result()
        training_client.optim_step(adam_params).result()

        # 4. Log metrics
        elapsed = time.time() - t_start
        mean_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        mean_kl = sum(batch_kl) / len(batch_kl) if batch_kl else 0.0
        metrics = {
            "batch": batch_idx,
            "reward_mean": round(mean_reward, 4),
            "reward_min": round(min(batch_rewards), 4) if batch_rewards else 0.0,
            "reward_max": round(max(batch_rewards), 4) if batch_rewards else 0.0,
            "kl_mean": round(mean_kl, 4),
            "n_datums": len(datums),
            "elapsed_s": round(elapsed, 1),
        }
        logger.info(
            f"Batch {batch_idx}/{n_batches}: reward={mean_reward:.3f} kl={mean_kl:.3f} "
            f"datums={len(datums)} time={elapsed:.1f}s"
        )
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()

    metrics_file.close()

    # Save final weights
    logger.info("Saving final checkpoint...")
    training_client.save_state(name="final").result()
    save_result = training_client.save_weights_for_sampler(name="prox-final").result()
    model_path = save_result.path
    logger.info(f"Final model saved at: {model_path}")

    with open(os.path.join(LOG_DIR, "model_path.txt"), "w") as f:
        f.write(model_path)

    browser.close()
    pw.stop()

    # Run eval via eval.py subprocess
    n_eval = int(os.environ.get("N_EVAL", 10))
    if n_eval > 0:
        import subprocess
        from datetime import datetime
        eval_name = datetime.now().strftime("train_%Y%m%d_%H%M%S")
        logger.info(f"\nRunning eval (n={n_eval}, name={eval_name})...")
        subprocess.run([
            sys.executable, os.path.join(os.path.dirname(__file__), "eval.py"),
            "--n", str(n_eval),
            "--model_path", model_path,
            "--name", eval_name,
        ])

    logger.info("Done!")


if __name__ == "__main__":
    main()
