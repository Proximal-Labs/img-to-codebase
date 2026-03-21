"""
Multi-image flow agent RL training: action sequence screenshots → interactive HTML.

Each rollout:
1. Model sees interleaved flow screenshots (up to 5 key states + action descriptions)
2. Generates interactive HTML/JS
3. We run the action sequence via Playwright, compute per-step SSIM
4. Show model the worst failing steps (gen vs ref side-by-side)
5. Model fixes
6. Final reward = avg SSIM across action steps

Uses Mind2Web data: real website tasks with action sequences + screenshots.
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
from difflib import SequenceMatcher

from tinker_cookbook import renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    MODEL, LORA_RANK, RENDERER_NAME, BATCH_SIZE, GROUP_SIZE, MAX_BATCHES,
    LR, KL_BETA, PPO_CLIP_LOW, PPO_CLIP_HIGH, SAVE_EVERY,
    IMG_SIZE, LOG_DIR,
)
from reward import (
    render_html, extract_html_from_response, make_diff_image,
    render_html_to_image, compute_reward_from_info,
    extract_ref_info, extract_gen_info,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TURNS = int(os.environ.get("MAX_TURNS", 2))
TOKENS_PER_TURN = int(os.environ.get("TOKENS_PER_TURN", 4096))
MAX_FLOW_IMAGES = 3
MAX_ACTION_STEPS = 3  # Cap action steps during training for speed
VIEWPORT = {"width": 1280, "height": 720}

SYSTEM_PROMPT = (
    "You are an expert at generating interactive HTML/CSS/JS websites. "
    "You will see screenshots from a user flow showing different states of a website. "
    "Generate a single HTML file that reproduces all the pages with working interactions. "
    "You may use Tailwind CSS, inline styles, or a <style> block. "
    "Wrap your code in ```html ... ```.\n\n"
    "After your first attempt, you will see which interaction steps failed "
    "with side-by-side comparisons. Fix the issues and resubmit."
)


def take_screenshot(page) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def compute_ssim(a, b):
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0])))
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def load_mind2web_tasks(n: int, seed: int = 42) -> list[dict]:
    """Load and group Mind2Web actions by task."""
    from datasets import load_dataset

    logger.info("Loading Mind2Web (streaming)...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train", streaming=True)

    tasks = {}
    for i, row in enumerate(ds):
        if len(tasks) >= n * 3:
            break
        if i > 10000:
            break

        ann_id = row["annotation_id"]
        if ann_id not in tasks:
            tasks[ann_id] = {
                "annotation_id": ann_id,
                "task": row["confirmed_task"],
                "website": row["website"],
                "actions": [],
            }

        op = json.loads(row["operation"]) if isinstance(row["operation"], str) else row["operation"]
        pos = json.loads(row["pos_candidates"][0]) if row["pos_candidates"] else {}
        if isinstance(pos, str):
            pos = json.loads(pos)
        attrs = pos.get("attributes", "{}") if isinstance(pos, dict) else "{}"
        if isinstance(attrs, str):
            attrs = json.loads(attrs)

        selector = f"#{attrs['id']}" if attrs.get("id") else None
        bbox = attrs.get("bounding_box_rect")

        screenshot = row["screenshot"]
        if screenshot:
            viewport_img = screenshot.crop((0, 0, min(screenshot.width, 1280), min(screenshot.height, 720)))
        else:
            viewport_img = None

        tasks[ann_id]["actions"].append({
            "op": op.get("op", "CLICK"),
            "value": op.get("value", ""),
            "selector": selector,
            "bbox": bbox,
            "repr": row["target_action_reprs"],
            "screenshot": viewport_img,
        })

    valid = [t for t in tasks.values() if len(t["actions"]) >= 2 and t["actions"][0]["screenshot"]]
    random.seed(seed)
    random.shuffle(valid)
    selected = valid[:n]
    logger.info(f"  Selected {len(selected)} tasks")
    return selected


def build_flow_prompt(actions, renderer):
    """Build interleaved flow screenshot + action description content."""
    steps = [(i, a) for i, a in enumerate(actions) if a.get("screenshot")]
    if len(steps) > MAX_FLOW_IMAGES:
        selected = [steps[0]]
        middle = steps[1:-1]
        step_size = max(1, len(middle) // (MAX_FLOW_IMAGES - 2))
        selected.extend(middle[::step_size][:MAX_FLOW_IMAGES - 2])
        selected.append(steps[-1])
    else:
        selected = steps

    convo_content = [{"type": "text", "text": "Here is a website user flow. Generate HTML/CSS/JS that reproduces this page with all interactions working.\n"}]

    for j, (idx, action) in enumerate(selected):
        desc = action["repr"]
        if j == 0:
            convo_content.append({"type": "text", "text": f"\nStep {j+1} — Initial page load:"})
        else:
            convo_content.append({"type": "text", "text": f"\nStep {j+1} — {desc}:"})

        img = action["screenshot"].resize((1280, 720))
        convo_content.append({"type": "image", "image": img})

    convo_content.append({"type": "text", "text": "\n\nGenerate complete HTML/CSS/JS with all pages and interactions. Wrap in ```html ... ```."})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": convo_content},
    ]


def run_actions_on_page(page, actions):
    """Run action sequence (capped at MAX_ACTION_STEPS), return per-step SSIM and screenshots."""
    results = []
    for i, action in enumerate(actions[:MAX_ACTION_STEPS]):
        ref_screenshot = action.get("screenshot")
        if ref_screenshot is None:
            continue

        gen_img = take_screenshot(page)
        ref_arr = np.array(ref_screenshot.resize((gen_img.shape[1], gen_img.shape[0])))
        ssim = compute_ssim(ref_arr, gen_img)

        results.append({"step": i, "ssim": ssim, "gen_img": gen_img, "ref_img": ref_arr, "action": action["repr"]})

        # Execute action
        op = action["op"]
        selector = action.get("selector")
        action_text = ""
        repr_str = action.get("repr", "")
        if "->" in repr_str:
            action_text = repr_str.split("->")[0].strip()
            if "]" in action_text:
                action_text = action_text.split("]", 1)[1].strip()

        if op == "CLICK":
            clicked = False
            if selector:
                try:
                    page.click(selector, timeout=2000)
                    clicked = True
                except Exception:
                    pass
            if not clicked and action_text:
                try:
                    page.get_by_text(action_text, exact=False).first.click(timeout=2000)
                    clicked = True
                except Exception:
                    pass
            if clicked:
                page.wait_for_timeout(500)

        elif op == "TYPE":
            typed = False
            if selector:
                try:
                    page.fill(selector, action.get("value", ""), timeout=2000)
                    typed = True
                except Exception:
                    pass
            if not typed and action_text:
                try:
                    page.get_by_placeholder(action_text).first.fill(action.get("value", ""), timeout=2000)
                    typed = True
                except Exception:
                    pass
            if typed:
                page.wait_for_timeout(300)

    return results


def compute_flow_reward(step_results):
    """Compute reward from step SSIM scores. 0.80 SSIM + 0.10 step_consistency + 0.10 coverage."""
    if not step_results:
        return -1.0
    ssims = [r["ssim"] for r in step_results]
    avg_ssim = np.mean(ssims)
    return float(2.0 * avg_ssim - 1.0)  # Scale to [-1, 1]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    n_tasks = int(os.environ.get("N_TASKS", 500))
    dataset = load_mind2web_tasks(n_tasks, seed=42)
    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"Loaded {len(dataset)} tasks, {n_batches} batches")
    logger.info(f"Flow agent: MAX_TURNS={MAX_TURNS}, TOKENS={TOKENS_PER_TURN}")

    tokenizer = get_tokenizer(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer, image_processor=image_processor)

    service_client = tinker.ServiceClient()
    resume_path = os.environ.get("RESUME_FROM")
    if resume_path:
        logger.info(f"Resuming from: {resume_path}")
        training_client = service_client.create_training_client_from_state_with_optimizer(resume_path)
    else:
        training_client = service_client.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)

    sampling_params = types.SamplingParams(
        max_tokens=TOKENS_PER_TURN,
        stop=renderer.get_stop_sequences(),
        temperature=0.7,
    )
    adam_params = types.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)

    metrics_path = os.path.join(LOG_DIR, "metrics_flow.jsonl")
    metrics_file = open(metrics_path, "a")

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    pages = [browser.new_page(viewport=VIEWPORT) for _ in range(BATCH_SIZE)]

    logger.info(f"Config: GROUP={GROUP_SIZE}, KL={KL_BETA}, TURNS={MAX_TURNS}, SAVE={SAVE_EVERY}")

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            logger.info(f"Saving checkpoint at batch {batch_idx}...")
            training_client.save_state(name=f"checkpoint-{batch_idx:04d}").result()

        sampling_client = training_client.save_weights_and_get_sampling_client()

        datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_kl: list[float] = []

        for idx in tqdm(range(len(batch)), desc=f"Batch {batch_idx}"):
            task = batch[idx]
            page = pages[idx % len(pages)]
            actions = task["actions"]

            # Build flow prompt
            convo = build_flow_prompt(actions, renderer)
            initial_prompt = renderer.build_generation_prompt(convo)

            # Parallel turn 1
            turn1_result = sampling_client.sample(
                prompt=initial_prompt, num_samples=GROUP_SIZE, sampling_params=sampling_params,
            ).result()

            rewards_G = []
            tokens_G = []
            logprobs_G = []

            for g, seq in enumerate(turn1_result.sequences):
                all_tokens = list(seq.tokens)
                all_logprobs = list(seq.logprobs)

                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                current_html = extract_html_from_response(content)
                final_reward = -1.0

                if current_html is not None:
                    # Run action sequence
                    render_html(page, current_html)
                    page.wait_for_timeout(500)
                    step_results = run_actions_on_page(page, actions)
                    final_reward = compute_flow_reward(step_results)

                    # Turn 2: show worst steps as feedback
                    if final_reward < 0.8 and MAX_TURNS > 1:
                        worst = sorted(step_results, key=lambda x: x["ssim"])[:3]

                        feedback = [{"type": "text", "text": (
                            f"Flow SSIM: {(final_reward + 1) / 2:.0%}. "
                            f"Top issues:\n"
                        )}]

                        for w in worst:
                            ref_pil = Image.fromarray(w["ref_img"])
                            gen_pil = Image.fromarray(w["gen_img"])
                            feedback.append({"type": "text", "text": f"\nStep {w['step']} — {w['action'][:40]} (SSIM={w['ssim']:.0%}):\nYour output:"})
                            feedback.append({"type": "image", "image": gen_pil})
                            feedback.append({"type": "text", "text": "Target:"})
                            feedback.append({"type": "image", "image": ref_pil})

                        feedback.append({"type": "text", "text": "\nList top 3 fixes needed. Be specific."})

                        convo.append({"role": "assistant", "content": content})
                        convo.append({"role": "user", "content": feedback})

                        # Analyze
                        analyze_prompt = renderer.build_generation_prompt(convo)
                        analyze_result = sampling_client.sample(
                            prompt=analyze_prompt, num_samples=1, sampling_params=sampling_params,
                        ).result()
                        analyze_seq = analyze_result.sequences[0]
                        all_tokens.extend(analyze_seq.tokens)
                        all_logprobs.extend(analyze_seq.logprobs)

                        analyze_msg, _ = renderer.parse_response(analyze_seq.tokens)
                        analysis = get_text_content(analyze_msg)

                        # Fix
                        convo.append({"role": "assistant", "content": analysis})
                        convo.append({"role": "user", "content": "Fix ALL issues. Output complete corrected HTML in ```html ... ```."})

                        fix_prompt = renderer.build_generation_prompt(convo)
                        fix_result = sampling_client.sample(
                            prompt=fix_prompt, num_samples=1, sampling_params=sampling_params,
                        ).result()
                        fix_seq = fix_result.sequences[0]
                        all_tokens.extend(fix_seq.tokens)
                        all_logprobs.extend(fix_seq.logprobs)

                        parsed_fix, _ = renderer.parse_response(fix_seq.tokens)
                        fix_content = get_text_content(parsed_fix)
                        fixed_html = extract_html_from_response(fix_content)

                        if fixed_html is not None:
                            render_html(page, fixed_html)
                            page.wait_for_timeout(500)
                            step_results2 = run_actions_on_page(page, actions)
                            final_reward = compute_flow_reward(step_results2)

                tokens_G.append(all_tokens)
                logprobs_G.append(all_logprobs)
                kl = -sum(all_logprobs) / len(all_logprobs) if all_logprobs else 0.0
                rewards_G.append(final_reward - KL_BETA * kl)
                batch_kl.append(kl)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            batch_rewards.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

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
        logger.info(f"Batch {batch_idx}/{n_batches}: reward={mean_reward:.3f} kl={mean_kl:.3f} datums={len(datums)} time={elapsed:.1f}s")
        metrics_file.write(json.dumps({"batch": batch_idx, "reward": round(mean_reward, 4), "kl": round(mean_kl, 4), "datums": len(datums), "time": round(elapsed, 1)}) + "\n")
        metrics_file.flush()

    metrics_file.close()

    logger.info("Saving final...")
    training_client.save_state(name="final").result()
    save_result = training_client.save_weights_for_sampler(name="prox-flow-final").result()
    logger.info(f"Saved: {save_result.path}")

    with open(os.path.join(LOG_DIR, "model_path_flow.txt"), "w") as f:
        f.write(save_result.path)

    browser.close()
    pw.stop()
    logger.info("Done!")


if __name__ == "__main__":
    main()
