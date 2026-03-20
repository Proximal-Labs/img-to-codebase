"""
Multi-turn agent RL training: screenshot → HTML/CSS with iterative self-correction.

Each rollout is up to MAX_TURNS attempts:
1. Model sees reference screenshot, generates HTML
2. We render it, create a diff image (red highlights where wrong)
3. Model sees diff image, generates improved HTML
4. Repeat until MAX_TURNS
5. Final HTML gets the reward

The model never sees the reward score — only visual diff feedback.
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
    compute_reward_from_info, extract_html_from_response,
    extract_ref_info, extract_gen_info, render_html_to_image,
    make_diff_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TURNS = int(os.environ.get("MAX_TURNS", 3))
TOKENS_PER_TURN = int(os.environ.get("TOKENS_PER_TURN", 2048))

SYSTEM_PROMPT_AGENT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "You may use Tailwind CSS, inline styles, or a <style> block. "
    "Wrap your code in ```html ... ```.\n\n"
    "After each attempt, you will receive a diff image where red-highlighted regions "
    "show where your output differs from the target. Use this feedback to improve "
    "your output — fix the red areas and resubmit."
)

def make_feedback_prompt(ssim_score: float, diff_pct: float) -> str:
    """Generate feedback prompt with SSIM score and diff coverage."""
    quality = "very close" if ssim_score > 0.8 else "partially matching" if ssim_score > 0.5 else "significantly different"
    return (
        f"Visual similarity score: {ssim_score:.0%} — your output is {quality} to the target.\n\n"
        f"The diff image above highlights problem areas in red ({diff_pct:.0%} of pixels differ). "
        f"Red regions indicate where your HTML renders differently from the target — "
        f"this could be wrong colors, missing elements, incorrect sizing, or layout issues.\n\n"
        f"Please fix the red areas and output the complete corrected HTML in ```html ... ```."
    )


def load_dataset(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        return json.load(f)


def load_training_data() -> list[dict]:
    dataset = load_dataset(MANIFEST_PATH)
    logger.info(f"  WebSight: {len(dataset)} examples")
    if os.path.exists(DESIGN2CODE_MANIFEST):
        from config import MAX_HTML_CHARS
        d2c = load_dataset(DESIGN2CODE_MANIFEST)
        d2c = [x for x in d2c if len(x.get("html", "")) < MAX_HTML_CHARS]
        logger.info(f"  Design2Code: {len(d2c)} examples (filtered)")
        dataset.extend(d2c)
    dataset.sort(key=lambda x: len(x.get("html", "")))
    return dataset


def run_agent_rollout(
    sampling_client,
    renderer,
    ref_pil: Image.Image,
    ref_info: dict,
    ref_render: np.ndarray,
    page,
    sampling_params,
) -> tuple[list[int], list[float], float]:
    """
    Run a single multi-turn agent rollout.

    Returns (all_tokens, all_logprobs, final_reward).
    Tokens/logprobs are concatenated across all turns.
    """
    all_tokens = []
    all_logprobs = []

    # Build initial prompt
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": ref_pil},
                {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
            ],
        },
    ]

    current_html = None
    final_reward = -1.0

    for turn in range(MAX_TURNS):
        prompt = renderer.build_generation_prompt(convo)

        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        seq = result.sequences[0]
        all_tokens.extend(seq.tokens)
        all_logprobs.extend(seq.logprobs)

        # Parse HTML from response
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = get_text_content(parsed_msg)
        current_html = extract_html_from_response(content)

        if current_html is None:
            # Model failed to produce HTML, stop
            break

        # Render and compute reward
        try:
            gen_info = extract_gen_info(page, current_html, size=IMG_SIZE)
            final_reward, _ = compute_reward_from_info(ref_info, gen_info)
        except Exception:
            final_reward = -1.0
            break

        # If reward is high enough or last turn, stop
        if final_reward > 0.9 or turn == MAX_TURNS - 1:
            break

        # Create diff at viewport resolution
        gen_render = render_html_to_image(page, current_html, size=max(VIEWPORT_W, VIEWPORT_H))
        diff_img = make_diff_image(ref_render, gen_render, threshold=25)
        diff_pil = Image.fromarray(diff_img)

        # Compute SSIM and diff percentage for feedback
        from skimage.metrics import structural_similarity as ssim_fn
        if ref_render.shape == gen_render.shape:
            ssim_score = ssim_fn(ref_render, gen_render, channel_axis=2, data_range=255)
        else:
            ssim_score = 0.0
        diff_mask = np.any(np.abs(ref_render.astype(int) - gen_render.astype(int)) > 25, axis=2)
        diff_pct = diff_mask.sum() / diff_mask.size

        feedback = make_feedback_prompt(ssim_score, diff_pct)

        # Add the model's response and feedback to conversation
        convo.append({"role": "assistant", "content": content})
        convo.append({
            "role": "user",
            "content": [
                {"type": "image", "image": diff_pil},
                {"type": "text", "text": feedback},
            ],
        })

    return all_tokens, all_logprobs, final_reward


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    dataset = load_training_data()
    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"Loaded {len(dataset)} examples, {n_batches} batches")
    logger.info(f"Agent mode: MAX_TURNS={MAX_TURNS}, TOKENS_PER_TURN={TOKENS_PER_TURN}")

    # Setup
    tokenizer = get_tokenizer(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer, image_processor=image_processor)

    service_client = tinker.ServiceClient()
    resume_path = os.environ.get("RESUME_FROM")
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        training_client = service_client.create_training_client_from_state_with_optimizer(resume_path)
    else:
        training_client = service_client.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)

    sampling_params = types.SamplingParams(
        max_tokens=TOKENS_PER_TURN,
        stop=renderer.get_stop_sequences(),
        temperature=0.7,
    )
    adam_params = types.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)

    metrics_path = os.path.join(LOG_DIR, "metrics.jsonl")
    metrics_file = open(metrics_path, "a")

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    reward_pages = [
        browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
        for _ in range(BATCH_SIZE)
    ]

    logger.info(f"Config: GROUP_SIZE={GROUP_SIZE}, KL_BETA={KL_BETA}, MAX_TURNS={MAX_TURNS}")

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            logger.info(f"Saving checkpoint at batch {batch_idx}...")
            training_client.save_state(name=f"checkpoint-{batch_idx:04d}").result()

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Pre-extract reference info + viewport-sized render for diffs
        ref_infos = []
        ref_pils = []
        ref_renders = []
        for i, item in enumerate(batch):
            page = reward_pages[i % len(reward_pages)]
            ref_html = item.get("reference_html") or item["html"]
            ref_info = extract_ref_info(page, ref_html, size=IMG_SIZE)
            ref_infos.append(ref_info)
            # Viewport-sized render for model input + diff feedback
            ref_render = render_html_to_image(page, ref_html, size=max(VIEWPORT_W, VIEWPORT_H))
            ref_renders.append(ref_render)
            ref_pils.append(Image.fromarray(ref_render))

        # Run multi-turn rollouts
        datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_kl: list[float] = []

        for idx in tqdm(range(len(batch)), desc=f"Batch {batch_idx}"):
            ref_info = ref_infos[idx]
            ref_pil = ref_pils[idx]
            ref_render = ref_renders[idx]
            page = reward_pages[idx % len(reward_pages)]

            # Run GROUP_SIZE rollouts for this prompt
            rewards_G = []
            tokens_G = []
            logprobs_G = []

            for g in range(GROUP_SIZE):
                all_tokens, all_logprobs, reward = run_agent_rollout(
                    sampling_client, renderer, ref_pil, ref_info, ref_render, page, sampling_params,
                )
                tokens_G.append(all_tokens)
                logprobs_G.append(all_logprobs)

                kl = -sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0
                rewards_G.append(reward - KL_BETA * kl)
                batch_kl.append(kl)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            batch_rewards.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            # Build training datums from the initial prompt + full trajectory
            initial_prompt = renderer.build_generation_prompt([
                {"role": "system", "content": SYSTEM_PROMPT_AGENT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ref_pil},
                        {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
                    ],
                },
            ])

            for tokens, logprobs, advantage in zip(tokens_G, logprobs_G, advantages_G):
                if not tokens:
                    continue
                ob_len = initial_prompt.length - 1
                model_input = initial_prompt.append(types.EncodedTextChunk(tokens=tokens[:-1]))
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

        # Training step
        if len(datums) == 0:
            logger.warning(f"Batch {batch_idx}: no datums, skipping")
            continue

        training_client.forward_backward(
            datums, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW, "clip_high_threshold": PPO_CLIP_HIGH},
        ).result()
        training_client.optim_step(adam_params).result()

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

    # Save
    logger.info("Saving final checkpoint...")
    training_client.save_state(name="final").result()
    save_result = training_client.save_weights_for_sampler(name="prox-agent-final").result()
    model_path = save_result.path
    logger.info(f"Final model saved at: {model_path}")

    with open(os.path.join(LOG_DIR, "model_path.txt"), "w") as f:
        f.write(model_path)

    browser.close()
    pw.stop()
    logger.info("Done!")


if __name__ == "__main__":
    main()
