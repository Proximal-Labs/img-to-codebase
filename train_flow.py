"""
Interactive flow RL training: action sequence screenshots → interactive HTML/JS.

Each rollout:
1. Model sees interleaved flow screenshots (up to 3 action steps + descriptions)
2. Generates interactive HTML/JS
3. We run action sequence via Playwright, compute per-step SSIM
4. Model sees target vs generated for worst steps, analyzes and fixes
5. Final reward = avg SSIM across action steps

Parallelism: turn 1 all parallel, turn 2 analyze all parallel, fix all parallel.
"""

import io
import json
import logging
import os
import random
import time

import numpy as np
import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from PIL import Image
from playwright.sync_api import sync_playwright
from skimage.metrics import structural_similarity as ssim_fn
from transformers import AutoProcessor

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    MODEL, LORA_RANK, BATCH_SIZE, GROUP_SIZE, MAX_BATCHES,
    LR, KL_BETA, PPO_CLIP_LOW, PPO_CLIP_HIGH, SAVE_EVERY, LOG_DIR,
)
from reward import render_html, extract_html_from_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TURNS = int(os.environ.get("MAX_TURNS", 2))
TOKENS_PER_TURN = int(os.environ.get("TOKENS_PER_TURN", 8192))
MAX_ACTION_STEPS = int(os.environ.get("MAX_ACTION_STEPS", 3))
NUM_PAGES = int(os.environ.get("NUM_PAGES", 32))
RENDERER_NAME = os.environ.get("RENDERER_NAME", "qwen3_disable_thinking")
VIEWPORT = {"width": 1280, "height": 720}

SYSTEM_PROMPT = (
    "You are an expert at generating interactive HTML/CSS/JS websites. "
    "You will see screenshots from a user flow showing different states of a website. "
    "Generate a single HTML file that reproduces all the pages with working interactions. "
    "You may use Tailwind CSS, inline styles, or a <style> block. "
    "Wrap your code in ```html ... ```.\n\n"
    "After your first attempt, you will see the target vs your output for the worst steps. "
    "Analyze what's different and fix your HTML."
)

# ── VLM prompt building ──────────────────────────────────────────────────────

IMAGE_PAD_TOKEN = 248056
_vlm_processor = None
_vlm_tokenizer = None


def init_vlm(processor, tokenizer):
    global _vlm_processor, _vlm_tokenizer
    _vlm_processor = processor
    _vlm_tokenizer = tokenizer


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_vlm_prompt(messages: list[dict]) -> types.ModelInput:
    proc, tok = _vlm_processor, _vlm_tokenizer

    images = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    images.append(part["image"])

    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    text = text.replace("<think>\n", "")
    token_ids = tok.encode(text, add_special_tokens=False)

    # Get expected token counts per image
    img_token_counts = []
    for img in images:
        test_msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": "x"},
        ]}]
        test_text = proc.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
        test_inputs = proc(text=[test_text], images=[img], return_tensors="pt")
        n = (test_inputs["input_ids"] == IMAGE_PAD_TOKEN).sum().item()
        img_token_counts.append(n)

    chunks = []
    current_tokens = []
    img_idx = 0
    for tok_id in token_ids:
        if tok_id == IMAGE_PAD_TOKEN:
            if current_tokens:
                chunks.append(types.EncodedTextChunk(tokens=current_tokens))
                current_tokens = []
            img = images[img_idx] if img_idx < len(images) else images[-1]
            expected = img_token_counts[img_idx] if img_idx < len(img_token_counts) else 64
            chunks.append(types.ImageChunk(
                data=pil_to_png_bytes(img), format="png", expected_tokens=expected,
            ))
            img_idx += 1
        else:
            current_tokens.append(tok_id)
    if current_tokens:
        chunks.append(types.EncodedTextChunk(tokens=current_tokens))

    return types.ModelInput(chunks=chunks)


def get_text_content(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return str(content)


# ── Mind2Web data loading ─────────────────────────────────────────────────────

def load_mind2web_tasks(n: int, seed: int = 42) -> list[dict]:
    """Load Mind2Web tasks with action sequences."""
    from datasets import load_dataset

    logger.info("Loading Mind2Web (streaming)...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train", streaming=True)

    tasks = {}
    for i, row in enumerate(ds):
        if len(tasks) >= n * 3:
            break
        if i > 15000:
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

        screenshot = row["screenshot"]
        if screenshot:
            viewport_img = screenshot.crop((0, 0, min(screenshot.width, 1280), min(screenshot.height, 720)))
        else:
            viewport_img = None

        tasks[ann_id]["actions"].append({
            "op": op.get("op", "CLICK"),
            "value": op.get("value", ""),
            "selector": selector,
            "repr": row["target_action_reprs"],
            "screenshot": viewport_img,
        })

    # Need at least 2 actions with screenshots
    valid = [t for t in tasks.values()
             if len(t["actions"]) >= 2
             and t["actions"][0].get("screenshot") is not None]
    random.seed(seed)
    random.shuffle(valid)
    selected = valid[:n]
    logger.info(f"  Selected {len(selected)} tasks with {sum(len(t['actions']) for t in selected)} total actions")
    return selected


# ── Action execution ──────────────────────────────────────────────────────────

def take_screenshot(page) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def compute_ssim(a, b):
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0])))
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def run_actions_on_page(page, actions, max_steps=None):
    """Run action sequence, return per-step results with SSIM and screenshots."""
    if max_steps is None:
        max_steps = MAX_ACTION_STEPS
    results = []
    for i, action in enumerate(actions[:max_steps]):
        ref_screenshot = action.get("screenshot")
        if ref_screenshot is None:
            continue

        gen_img = take_screenshot(page)
        ref_arr = np.array(ref_screenshot.resize((gen_img.shape[1], gen_img.shape[0])))
        ssim = compute_ssim(ref_arr, gen_img)

        results.append({
            "step": i, "ssim": ssim,
            "gen_img": gen_img, "ref_img": ref_arr,
            "action": action["repr"],
        })

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
                page.wait_for_timeout(100)

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
                page.wait_for_timeout(100)

    return results


def compute_flow_reward(step_results):
    if not step_results:
        return -1.0
    ssims = [r["ssim"] for r in step_results]
    return float(2.0 * np.mean(ssims) - 1.0)


# ── Flow prompt building ─────────────────────────────────────────────────────

def build_flow_messages(actions):
    """Build interleaved flow screenshots + action descriptions."""
    steps = [(i, a) for i, a in enumerate(actions) if a.get("screenshot")]
    selected = steps[:MAX_ACTION_STEPS + 1]  # +1 for initial page

    content = [{"type": "text", "text": (
        "Here is a website user flow. Generate HTML/CSS/JS that reproduces "
        "this page with all interactions working.\n"
    )}]

    for j, (idx, action) in enumerate(selected):
        if j == 0:
            content.append({"type": "text", "text": f"\nStep {j+1} — Initial page load:"})
        else:
            content.append({"type": "text", "text": f"\nStep {j+1} — {action['repr']}:"})
        content.append({"type": "image", "image": action["screenshot"]})

    content.append({"type": "text", "text": (
        "\n\nGenerate complete HTML/CSS/JS with all pages and interactions. "
        "Wrap in ```html ... ```."
    )})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


# ── Main training loop ───────────────────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    n_tasks = int(os.environ.get("N_TASKS", 500))
    dataset = load_mind2web_tasks(n_tasks, seed=42)
    n_batches = len(dataset) // BATCH_SIZE
    if MAX_BATCHES > 0:
        n_batches = min(n_batches, MAX_BATCHES)
    logger.info(f"Loaded {len(dataset)} tasks, {n_batches} batches")
    logger.info(f"Flow: MAX_TURNS={MAX_TURNS}, TOKENS={TOKENS_PER_TURN}, MAX_ACTIONS={MAX_ACTION_STEPS}")

    tokenizer = get_tokenizer(MODEL)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    init_vlm(processor, tokenizer)

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
    pages = [browser.new_page(viewport=VIEWPORT) for _ in range(NUM_PAGES)]

    logger.info(f"Config: BS={BATCH_SIZE}, GS={GROUP_SIZE}, KL={KL_BETA}, PAGES={NUM_PAGES}")

    pending_train_futures = None

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            if pending_train_futures:
                pending_train_futures[0].result()
                pending_train_futures[1].result()
                pending_train_futures = None
            logger.info(f"Saving checkpoint at batch {batch_idx}...")
            training_client.save_state(name=f"checkpoint-{batch_idx:04d}").result()

        sampling_client = training_client.save_weights_and_get_sampling_client()

        # ── Turn 1: fire ALL samples in parallel ─────────────────────────
        turn1_futures = []
        initial_prompts = []
        batch_actions = []

        for idx in range(len(batch)):
            task = batch[idx]
            actions = task["actions"]
            batch_actions.append(actions)
            convo = build_flow_messages(actions)
            prompt = build_vlm_prompt(convo)
            initial_prompts.append(prompt)
            turn1_futures.append(sampling_client.sample(
                prompt=prompt, num_samples=GROUP_SIZE, sampling_params=sampling_params,
            ))

        # Collect turn 1 results + run actions + compute rewards
        rollouts = []
        for idx in range(len(batch)):
            result = turn1_futures[idx].result()
            actions = batch_actions[idx]

            for g, seq in enumerate(result.sequences):
                flat_idx = len(rollouts)
                page = pages[flat_idx % len(pages)]

                tokens = list(seq.tokens)
                logprobs = list(seq.logprobs)
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                html = extract_html_from_response(content)
                reward = -1.0
                step_results = []

                if html is not None:
                    try:
                        render_html(page, html)
                        page.wait_for_timeout(100)
                        step_results = run_actions_on_page(page, actions)
                        reward = compute_flow_reward(step_results)
                    except Exception:
                        pass

                rollouts.append({
                    "tokens": tokens, "logprobs": logprobs,
                    "html": html, "content": content, "reward": reward,
                    "step_results": step_results,
                    "done": html is None or reward > 0.9,
                    "idx": idx, "g": g,
                    "convo": build_flow_messages(actions),
                })

        logger.info(f"  Turn 1 done: {sum(1 for r in rollouts if r['done'])}/{len(rollouts)} done, "
                     f"avg reward={np.mean([r['reward'] for r in rollouts]):.3f}")

        # ── Turn 2+: parallel across ALL active rollouts ─────────────────
        for turn in range(1, MAX_TURNS):
            active = [r for r in rollouts if not r["done"]]
            if not active:
                break
            logger.info(f"  Turn {turn+1}: {len(active)} active rollouts")

            # Phase A: build feedback + fire analyze calls in parallel
            analyze_futures = []
            for r in active:
                idx = r["idx"]
                worst = sorted(r["step_results"], key=lambda x: x["ssim"])[:3]

                feedback = [{"type": "text", "text": (
                    f"Your HTML renders but doesn't fully match. Here are the worst steps:\n"
                )}]
                for w in worst:
                    ref_pil = Image.fromarray(w["ref_img"])
                    gen_pil = Image.fromarray(w["gen_img"])
                    feedback.append({"type": "text", "text": f"\nStep {w['step']+1} — {w['action'][:60]}:"})
                    feedback.append({"type": "text", "text": "Target:"})
                    feedback.append({"type": "image", "image": ref_pil})
                    feedback.append({"type": "text", "text": "Your output:"})
                    feedback.append({"type": "image", "image": gen_pil})

                feedback.append({"type": "text", "text": (
                    "\nCompare target vs your output. List the specific visual differences "
                    "and what needs to change in the HTML/CSS/JS. Be concise."
                )})

                r["convo"].append({"role": "assistant", "content": r["content"]})
                r["convo"].append({"role": "user", "content": feedback})

                analyze_prompt = build_vlm_prompt(r["convo"])
                analyze_futures.append(sampling_client.sample(
                    prompt=analyze_prompt, num_samples=1, sampling_params=sampling_params,
                ))

            # Collect ALL analyze results
            for r, future in zip(active, analyze_futures):
                result = future.result()
                seq = result.sequences[0]
                r["tokens"].extend(seq.tokens)
                r["logprobs"].extend(seq.logprobs)

                parsed_msg, _ = renderer.parse_response(seq.tokens)
                analysis = get_text_content(parsed_msg)

                r["convo"].append({"role": "assistant", "content": analysis})
                r["convo"].append({"role": "user", "content": (
                    "Fix ALL issues. Output complete corrected HTML/CSS/JS in ```html ... ```."
                )})

            # Phase B: fire ALL fix calls in parallel
            fix_futures = []
            for r in active:
                fix_prompt = build_vlm_prompt(r["convo"])
                fix_futures.append(sampling_client.sample(
                    prompt=fix_prompt, num_samples=1, sampling_params=sampling_params,
                ))

            # Collect ALL fix results + re-run actions
            for r, future in zip(active, fix_futures):
                result = future.result()
                seq = result.sequences[0]
                r["tokens"].extend(seq.tokens)
                r["logprobs"].extend(seq.logprobs)

                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                html = extract_html_from_response(content)

                r["content"] = content
                r["html"] = html

                if html is None:
                    r["done"] = True
                    continue

                try:
                    page = pages[rollouts.index(r) % len(pages)]
                    render_html(page, html)
                    page.wait_for_timeout(100)
                    step_results = run_actions_on_page(page, batch_actions[r["idx"]])
                    r["reward"] = compute_flow_reward(step_results)
                    r["step_results"] = step_results
                except Exception:
                    r["reward"] = -1.0
                    r["done"] = True
                    continue

                if r["reward"] > 0.9:
                    r["done"] = True

        # ── Build training datums ────────────────────────────────────────
        datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_kl: list[float] = []

        for idx in range(len(batch)):
            item_rollouts = [r for r in rollouts if r["idx"] == idx]
            rewards_G = []
            tokens_G = []
            logprobs_G = []

            for r in item_rollouts:
                kl = -sum(r["logprobs"]) / len(r["logprobs"]) if r["logprobs"] else 0.0
                rewards_G.append(r["reward"] - KL_BETA * kl)
                tokens_G.append(r["tokens"])
                logprobs_G.append(r["logprobs"])
                batch_kl.append(kl)

            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            batch_rewards.append(mean_reward)

            if all(a == 0.0 for a in advantages_G):
                continue

            initial_prompt = initial_prompts[idx]
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

        if pending_train_futures:
            pending_train_futures[0].result()
            pending_train_futures[1].result()
            pending_train_futures = None

        if len(datums) == 0:
            logger.warning(f"Batch {batch_idx}: no datums, skipping")
            continue

        fwd_bwd_future = training_client.forward_backward(
            datums, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW, "clip_high_threshold": PPO_CLIP_HIGH},
        )
        optim_future = training_client.optim_step(adam_params)
        pending_train_futures = (fwd_bwd_future, optim_future)

        elapsed = time.time() - t_start
        mean_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        mean_kl = sum(batch_kl) / len(batch_kl) if batch_kl else 0.0
        logger.info(f"Batch {batch_idx}/{n_batches}: reward={mean_reward:.3f} kl={mean_kl:.3f} "
                     f"datums={len(datums)} time={elapsed:.1f}s")
        metrics_file.write(json.dumps({
            "batch": batch_idx, "reward": round(mean_reward, 4),
            "kl": round(mean_kl, 4), "datums": len(datums), "time": round(elapsed, 1),
        }) + "\n")
        metrics_file.flush()

    metrics_file.close()

    if pending_train_futures:
        pending_train_futures[0].result()
        pending_train_futures[1].result()

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
