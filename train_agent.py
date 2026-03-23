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

import io
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
from transformers import AutoProcessor

from tinker_cookbook import renderers
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
)


def get_text_content(msg: dict) -> str:
    """Extract text content from a parsed message dict."""
    content = msg.get("content", "")
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
    return str(content)


IMAGE_PAD_TOKEN = 248056  # <|image_pad|> for Qwen3.5


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_vlm_processor = None
_vlm_tokenizer = None
_vlm_disable_thinking = True


def init_vlm(processor, tokenizer, disable_thinking=True):
    global _vlm_processor, _vlm_tokenizer, _vlm_disable_thinking
    _vlm_processor = processor
    _vlm_tokenizer = tokenizer
    _vlm_disable_thinking = disable_thinking


def build_vlm_prompt(messages: list[dict]) -> types.ModelInput:
    """
    Build a tinker ModelInput with interleaved text and image chunks
    for Qwen3.5 VLM models.
    """
    proc, tok = _vlm_processor, _vlm_tokenizer

    # Collect all images in order
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    images.append(part["image"])

    # Get tokenized text with image placeholders
    text = proc.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if _vlm_disable_thinking:
        text = text.replace("<think>\n", "")

    token_ids = tok.encode(text, add_special_tokens=False)

    # Count expected image tokens per image
    img_token_counts = []
    for img in images:
        test_msgs = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": "x"},
        ]}]
        test_text = proc.apply_chat_template(test_msgs, tokenize=False, add_generation_prompt=True)
        test_inputs = proc(text=[test_text], images=[img], return_tensors="pt")
        n_img_tokens = (test_inputs["input_ids"] == IMAGE_PAD_TOKEN).sum().item()
        img_token_counts.append(n_img_tokens)

    # Split tokens into chunks around IMAGE_PAD_TOKEN, insert ImageChunks
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
                data=pil_to_png_bytes(img),
                format="png",
                expected_tokens=expected,
            ))
            img_idx += 1
        else:
            current_tokens.append(tok_id)
    if current_tokens:
        chunks.append(types.EncodedTextChunk(tokens=current_tokens))

    return types.ModelInput(chunks=chunks)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TURNS = int(os.environ.get("MAX_TURNS", 3))
TOKENS_PER_TURN = int(os.environ.get("TOKENS_PER_TURN", 2048))
USE_THINKING = os.environ.get("USE_THINKING", "0") == "1"
NUM_PAGES = int(os.environ.get("NUM_PAGES", 32))  # Playwright pages for parallel rendering

SYSTEM_PROMPT_AGENT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "You may use Tailwind CSS, inline styles, or a <style> block. "
    "Wrap your code in ```html ... ```.\n\n"
    "After each attempt, you will see the target screenshot alongside your rendered output. "
    "Analyze the visual differences and fix your HTML to better match the target."
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
    """Load training data from MANIFEST_PATH."""
    dataset = load_dataset(MANIFEST_PATH)
    random.shuffle(dataset)
    logger.info(f"  Loaded {len(dataset)} examples from {MANIFEST_PATH}")
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
        prompt = build_vlm_prompt(convo)

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

        # Step 1: Show ref + generated + diff, ask model to ANALYZE
        gen_pil = Image.fromarray(gen_render)
        convo.append({"role": "assistant", "content": content})
        convo.append({
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"Visual similarity: {ssim_score:.0%} ({diff_pct:.0%} of pixels differ).\n\n"
                    f"Here is the target (what it should look like):"
                )},
                {"type": "image", "image": ref_pil},
                {"type": "text", "text": "Here is your output:"},
                {"type": "image", "image": gen_pil},
                {"type": "text", "text": (
                    "Here is the diff (red = where your output differs from the target):"
                )},
                {"type": "image", "image": diff_pil},
                {"type": "text", "text": (
                    "List the top 3 things that need to be fixed. Be specific — "
                    "e.g. 'heading font size is too large', 'nav background should be #333', "
                    "'missing footer section'. Be concise."
                )},
            ],
        })

        # Sample analysis
        analyze_prompt = build_vlm_prompt(convo)
        analyze_result = sampling_client.sample(
            prompt=analyze_prompt, num_samples=1, sampling_params=sampling_params,
        ).result()
        analyze_seq = analyze_result.sequences[0]
        all_tokens.extend(analyze_seq.tokens)
        all_logprobs.extend(analyze_seq.logprobs)

        analyze_msg, _ = renderer.parse_response(analyze_seq.tokens)
        analysis = get_text_content(analyze_msg)

        # Step 2: Ask model to FIX
        convo.append({"role": "assistant", "content": analysis})
        convo.append({"role": "user", "content": (
            "Now fix ALL the issues you identified. "
            "Output the complete corrected HTML in ```html ... ```."
        )})

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
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    renderer_name = "qwen3" if USE_THINKING else RENDERER_NAME
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    init_vlm(processor, tokenizer, disable_thinking=not USE_THINKING)
    logger.info(f"Thinking: {'enabled' if USE_THINKING else 'disabled'} (renderer: {renderer_name})")

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
    # Create enough pages for parallel rendering across all rollouts
    pages = [
        browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})
        for _ in range(NUM_PAGES)
    ]

    logger.info(f"Config: GROUP_SIZE={GROUP_SIZE}, KL_BETA={KL_BETA}, MAX_TURNS={MAX_TURNS}, NUM_PAGES={NUM_PAGES}")

    # Pipeline: get first sampling client, then overlap training with next batch's sampling
    pending_train_futures = None  # (fwd_bwd_future, optim_future) from previous batch

    for batch_idx in range(n_batches):
        t_start = time.time()
        batch = dataset[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        if SAVE_EVERY > 0 and batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
            # Must wait for previous training to finish before checkpointing
            if pending_train_futures:
                pending_train_futures[0].result()
                pending_train_futures[1].result()
                pending_train_futures = None
            logger.info(f"Saving checkpoint at batch {batch_idx}...")
            training_client.save_state(name=f"checkpoint-{batch_idx:04d}").result()

        # Get sampling client — if we have pending training, submit before waiting
        # so save_weights can be queued on the same clock cycle as optim_step
        sampling_client = training_client.save_weights_and_get_sampling_client()

        # Pre-extract reference info + viewport-sized render for diffs
        ref_infos = []
        ref_pils = []
        ref_renders = []
        for i, item in enumerate(batch):
            page = pages[i % len(pages)]
            ref_html = item.get("reference_html") or item["html"]
            ref_info = extract_ref_info(page, ref_html, size=IMG_SIZE)
            ref_infos.append(ref_info)
            ref_render = render_html_to_image(page, ref_html, size=max(VIEWPORT_W, VIEWPORT_H))
            ref_renders.append(ref_render)
            ref_pils.append(Image.fromarray(ref_render))

        # ── Turn 1: fire ALL samples in parallel ─────────────────────────
        turn1_futures = []
        initial_prompts = []
        for idx in range(len(batch)):
            prompt = build_vlm_prompt([
                {"role": "system", "content": SYSTEM_PROMPT_AGENT},
                {"role": "user", "content": [
                    {"type": "image", "image": ref_pils[idx]},
                    {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
                ]},
            ])
            initial_prompts.append(prompt)
            turn1_futures.append(sampling_client.sample(
                prompt=prompt, num_samples=GROUP_SIZE, sampling_params=sampling_params,
            ))

        # Collect turn 1 results and compute rewards for all rollouts
        # rollouts[flat_idx] = {tokens, logprobs, html, content, reward, done, convo, idx, g}
        rollouts = []
        for idx in range(len(batch)):
            result = turn1_futures[idx].result()
            for g, seq in enumerate(result.sequences):
                flat_idx = len(rollouts)
                page = pages[flat_idx % len(pages)]

                tokens = list(seq.tokens)
                logprobs = list(seq.logprobs)
                parsed_msg, _ = renderer.parse_response(seq.tokens)
                content = get_text_content(parsed_msg)
                html = extract_html_from_response(content)
                reward = -1.0

                if html is not None:
                    try:
                        gen_info = extract_gen_info(page, html, size=IMG_SIZE)
                        reward, _ = compute_reward_from_info(ref_infos[idx], gen_info)
                    except Exception:
                        pass

                rollouts.append({
                    "tokens": tokens, "logprobs": logprobs,
                    "html": html, "content": content, "reward": reward,
                    "done": html is None or reward > 0.9,
                    "idx": idx, "g": g,
                    "convo": [
                        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
                        {"role": "user", "content": [
                            {"type": "image", "image": ref_pils[idx]},
                            {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
                        ]},
                    ],
                })

        logger.info(f"  Turn 1 done: {sum(1 for r in rollouts if r['done'])}/{len(rollouts)} already done")

        # ── Turns 2+: parallel across ALL active rollouts ────────────────

        for turn in range(1, MAX_TURNS):
            active = [r for r in rollouts if not r["done"]]
            if not active:
                break
            logger.info(f"  Turn {turn+1}: {len(active)} active rollouts")

            # Phase A: render gen screenshots + fire analyze calls in parallel
            analyze_futures = []
            for r in active:
                page = pages[rollouts.index(r) % len(pages)]
                idx = r["idx"]

                gen_render = render_html_to_image(page, r["html"], size=max(VIEWPORT_W, VIEWPORT_H))
                gen_pil = Image.fromarray(gen_render)

                r["convo"].append({"role": "assistant", "content": r["content"]})
                r["convo"].append({"role": "user", "content": [
                    {"type": "text", "text": "Here is the target screenshot:"},
                    {"type": "image", "image": ref_pils[idx]},
                    {"type": "text", "text": "Here is what your HTML currently renders as:"},
                    {"type": "image", "image": gen_pil},
                    {"type": "text", "text": (
                        "Compare the two images. List the specific visual differences — "
                        "wrong colors, missing elements, incorrect sizing, layout issues, "
                        "wrong text. Be concise and specific."
                    )},
                ]})

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
                    "Now fix ALL the issues you identified. "
                    "Output the complete corrected HTML in ```html ... ```."
                )})

            # Phase B: fire ALL fix calls in parallel
            fix_futures = []
            for r in active:
                fix_prompt = build_vlm_prompt(r["convo"])
                fix_futures.append(sampling_client.sample(
                    prompt=fix_prompt, num_samples=1, sampling_params=sampling_params,
                ))

            # Collect ALL fix results + compute rewards
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
                    gen_info = extract_gen_info(page, html, size=IMG_SIZE)
                    r["reward"], _ = compute_reward_from_info(ref_infos[r["idx"]], gen_info)
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

        # Group rollouts back by batch item
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

        # Wait for previous batch's training to finish (if any)
        if pending_train_futures:
            pending_train_futures[0].result()
            pending_train_futures[1].result()
            pending_train_futures = None

        # Training step — pipeline fwd_bwd + optim_step on same clock cycle
        if len(datums) == 0:
            logger.warning(f"Batch {batch_idx}: no datums, skipping")
            continue

        fwd_bwd_future = training_client.forward_backward(
            datums, loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": PPO_CLIP_LOW, "clip_high_threshold": PPO_CLIP_HIGH},
        )
        optim_future = training_client.optim_step(adam_params)
        # Don't wait — let next batch's sampling overlap with training
        pending_train_futures = (fwd_bwd_future, optim_future)

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

    # Drain any pending training futures
    if pending_train_futures:
        pending_train_futures[0].result()
        pending_train_futures[1].result()

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
