"""
Eval multi-turn agent: model generates HTML, gets diff feedback, iterates.

Works with both Tinker models (our RL'd model) and OpenAI models.

Usage:
    # Our model (base Qwen3.5-4B)
    python eval_agent.py --n 5 --turns 3 --provider tinker

    # OpenAI
    python eval_agent.py --n 5 --turns 3 --provider openai --openai_model gpt-4.1

    # Our RL-trained model
    python eval_agent.py --n 5 --turns 3 --provider tinker --model_path "tinker://..."

    # Named output
    python eval_agent.py --n 5 --turns 3 --provider openai --name gpt4o-3turns
"""

import argparse
import base64
import io
import json

import os
import random

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from playwright.sync_api import sync_playwright

from config import (
    MODEL, LORA_RANK, RENDERER_NAME, IMG_SIZE, VIEWPORT_W, VIEWPORT_H,
    MANIFEST_PATH, EVAL_DIR, SYSTEM_PROMPT,
)
from reward import (
    extract_ref_info, extract_gen_info, extract_html_from_response,
    compute_reward_from_info, make_diff_image, render_html, render_html_to_file,
    render_html_to_image,
)
from train_agent import SYSTEM_PROMPT_AGENT


def log(msg):
    print(msg, flush=True)


def _viewport_screenshot(page, html: str) -> np.ndarray:
    """Render HTML and take a viewport-sized screenshot (e.g. 1024x768)."""
    render_html(page, html)
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── OpenAI agent ──────────────────────────────────────────────────────────────

ANALYZE_PROMPT = (
    "Look at the diff image above. The red-highlighted areas show where the generated HTML "
    "differs from the target screenshot.\n\n"
    "List the specific visual differences you can identify (e.g. wrong background color on nav, "
    "missing border-radius on button, text alignment is centered instead of left, etc). "
    "Be specific and concise — just list the issues, don't generate code yet."
)

FIX_PROMPT = (
    "Now fix ALL of the issues you identified above. "
    "Output the complete corrected HTML in ```html ... ```."
)


def run_openai_agent(
    client, model: str, ref_pil: Image.Image, ref_render: np.ndarray,
    ref_info: dict, page, max_turns: int,
) -> list[dict]:
    """Run multi-turn agent with OpenAI API. Returns list of turn results."""
    turns = []
    ref_b64 = pil_to_base64(ref_pil)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
            {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
        ]},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=4096, temperature=0.3,
        )
        content = response.choices[0].message.content
        html = extract_html_from_response(content)

        turn_result = {"turn": turn + 1, "html": html, "reward": -1.0, "ssim": 0.0, "diff_pct": 1.0}

        if html is None:
            turns.append(turn_result)
            break

        try:
            gen_info = extract_gen_info(page, html, size=IMG_SIZE)
            reward, details = compute_reward_from_info(ref_info, gen_info)
            gen_render = _viewport_screenshot(page, html)

            if ref_render.shape == gen_render.shape:
                ssim_score = ssim_fn(ref_render, gen_render, channel_axis=2, data_range=255)
            else:
                ssim_score = 0.0

            diff_mask = np.any(np.abs(ref_render.astype(int) - gen_render.astype(int)) > 25, axis=2)
            diff_pct = diff_mask.sum() / diff_mask.size

            turn_result.update({"reward": reward, "ssim": ssim_score, "diff_pct": diff_pct, "details": details})
        except Exception as e:
            turn_result["error"] = str(e)
            turns.append(turn_result)
            break

        turns.append(turn_result)

        if reward > 0.9 or turn == max_turns - 1:
            break

        # Step 1: Show diff and ask model to ANALYZE what's wrong
        diff_img = make_diff_image(ref_render, gen_render, threshold=25)
        diff_b64 = pil_to_base64(Image.fromarray(diff_img))

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{diff_b64}"}},
            {"type": "text", "text": (
                f"Visual similarity: {ssim_score:.0%} ({diff_pct:.0%} of pixels differ).\n\n"
                f"Above: the target screenshot (left) and the diff image (right) where "
                f"red areas show where your output differs.\n\n" + ANALYZE_PROMPT
            )},
        ]})

        # Get analysis
        analysis_response = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=1024, temperature=0.3,
        )
        analysis = analysis_response.choices[0].message.content
        turn_result["analysis"] = analysis

        # Step 2: Ask model to FIX based on its own analysis
        messages.append({"role": "assistant", "content": analysis})
        messages.append({"role": "user", "content": FIX_PROMPT})

    return turns


# ── Tinker agent ──────────────────────────────────────────────────────────────

def run_tinker_agent(
    sampling_client, renderer, ref_pil: Image.Image, ref_render: np.ndarray,
    ref_info: dict, page, max_turns: int, sampling_params,
) -> list[dict]:
    """Run multi-turn agent with Tinker. Returns list of turn results."""
    from train_agent import build_vlm_prompt, get_text_content
    turns = []

    convo = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT},
        {"role": "user", "content": [
            {"type": "image", "image": ref_pil},
            {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
        ]},
    ]

    for turn in range(max_turns):
        prompt = build_vlm_prompt(convo)
        result = sampling_client.sample(prompt=prompt, num_samples=1, sampling_params=sampling_params).result()

        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        content = get_text_content(parsed_msg)
        html = extract_html_from_response(content)

        turn_result = {"turn": turn + 1, "html": html, "reward": -1.0, "ssim": 0.0, "diff_pct": 1.0}

        if html is None:
            turns.append(turn_result)
            break

        try:
            gen_info = extract_gen_info(page, html, size=IMG_SIZE)
            reward, details = compute_reward_from_info(ref_info, gen_info)
            gen_render = _viewport_screenshot(page, html)

            if ref_render.shape == gen_render.shape:
                ssim_score = ssim_fn(ref_render, gen_render, channel_axis=2, data_range=255)
            else:
                ssim_score = 0.0

            diff_mask = np.any(np.abs(ref_render.astype(int) - gen_render.astype(int)) > 25, axis=2)
            diff_pct = diff_mask.sum() / diff_mask.size

            turn_result.update({"reward": reward, "ssim": ssim_score, "diff_pct": diff_pct, "details": details})
        except Exception as e:
            turn_result["error"] = str(e)
            turns.append(turn_result)
            break

        turns.append(turn_result)

        if reward > 0.9 or turn == max_turns - 1:
            break

        # Step 1: Show diff + ref, ask model to ANALYZE
        diff_img = make_diff_image(ref_render, gen_render, threshold=25)
        diff_pil = Image.fromarray(diff_img)

        convo.append({"role": "assistant", "content": content})
        convo.append({"role": "user", "content": [
            {"type": "image", "image": ref_pil},
            {"type": "image", "image": diff_pil},
            {"type": "text", "text": (
                f"Visual similarity: {ssim_score:.0%} ({diff_pct:.0%} of pixels differ).\n\n"
                f"Above: the target screenshot and the diff image where "
                f"red areas show where your output differs.\n\n" + ANALYZE_PROMPT
            )},
        ]})

        # Get analysis
        analyze_prompt = build_vlm_prompt(convo)
        analyze_result = sampling_client.sample(
            prompt=analyze_prompt, num_samples=1, sampling_params=sampling_params,
        ).result()
        analyze_msg, _ = renderer.parse_response(analyze_result.sequences[0].tokens)
        analysis = get_text_content(analyze_msg)
        turn_result["analysis"] = analysis

        # Step 2: Ask model to FIX
        convo.append({"role": "assistant", "content": analysis})
        convo.append({"role": "user", "content": FIX_PROMPT})

    return turns


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--provider", type=str, default="tinker", choices=["tinker", "openai"])
    parser.add_argument("--openai_model", type=str, default="gpt-4.1")
    parser.add_argument("--model_path", type=str, default=None, help="Tinker model path for RL weights")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    from datetime import datetime
    eval_name = args.name or f"agent_{args.provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_dir = os.path.join(EVAL_DIR, eval_name)
    os.makedirs(eval_dir, exist_ok=True)
    log(f"Eval dir: {eval_dir}")

    with open(MANIFEST_PATH) as f:
        dataset = json.load(f)
    samples = random.sample(dataset, min(args.n, len(dataset)))

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

    # Setup provider
    if args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        log(f"Using OpenAI: {args.openai_model}")
        sampling_client = None
        renderer = None
        sampling_params = None
    else:
        import tinker
        from tinker import types
        from tinker_cookbook import renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        from transformers import AutoProcessor
        from train_agent import init_vlm

        tokenizer = get_tokenizer(MODEL)
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
        init_vlm(processor, tokenizer)
        sampling_params = types.SamplingParams(
            max_tokens=4096, stop=renderer.get_stop_sequences(), temperature=0.3,
        )

        sc = tinker.ServiceClient()
        if args.model_path:
            log(f"Using Tinker RL model: {args.model_path}")
            sampling_client = sc.create_sampling_client(model_path=args.model_path)
        else:
            log(f"Using Tinker base model: {MODEL}")
            tc = sc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
            sampling_client = tc.save_weights_and_get_sampling_client()
        client = None

    all_results = []

    if args.provider == "tinker":
        # ── Parallel Tinker eval ─────────────────────────────────────────
        from train_agent import build_vlm_prompt, get_text_content

        # Prep all examples
        ex_dirs = []
        ref_infos = []
        ref_pils = []
        ref_renders = []
        for i, item in enumerate(samples):
            ex_dir = os.path.join(eval_dir, f"example_{i:02d}")
            os.makedirs(ex_dir, exist_ok=True)
            ex_dirs.append(ex_dir)

            ref_html = item.get("reference_html") or item["html"]
            if item.get("screenshot") and os.path.exists(item["screenshot"]):
                ref_pil_orig = Image.open(item["screenshot"]).convert("RGB")
                ref_render = np.array(ref_pil_orig.resize((VIEWPORT_W, VIEWPORT_H)))
                ref_pil = Image.fromarray(ref_render)
                ref_info = extract_ref_info(page, ref_html, size=IMG_SIZE)
                ref_info["image"] = np.array(ref_pil_orig.resize((IMG_SIZE, IMG_SIZE)))
            else:
                ref_info = extract_ref_info(page, ref_html, size=IMG_SIZE)
                render_html(page, ref_html)
                ref_render = np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
                ref_pil = Image.fromarray(ref_render)

            ref_pil.save(os.path.join(ex_dir, "ref-render.png"))
            ref_infos.append(ref_info)
            ref_pils.append(ref_pil)
            ref_renders.append(ref_render)

        # Turn 1: fire ALL in parallel
        log(f"\nFiring {len(samples)} turn-1 prompts in parallel...")
        turn1_futures = []
        for i in range(len(samples)):
            prompt = build_vlm_prompt([
                {"role": "system", "content": SYSTEM_PROMPT_AGENT},
                {"role": "user", "content": [
                    {"type": "image", "image": ref_pils[i]},
                    {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
                ]},
            ])
            turn1_futures.append(sampling_client.sample(
                prompt=prompt, num_samples=1, sampling_params=sampling_params,
            ))

        # Collect turn 1 results
        eval_states = []  # per-example state for multi-turn
        for i in range(len(samples)):
            result = turn1_futures[i].result()
            seq = result.sequences[0]
            parsed_msg, _ = renderer.parse_response(seq.tokens)
            content = get_text_content(parsed_msg)
            html = extract_html_from_response(content)

            turn_result = {"turn": 1, "html": html, "reward": -1.0, "ssim": 0.0, "diff_pct": 1.0}

            if html is not None:
                try:
                    gen_info = extract_gen_info(page, html, size=IMG_SIZE)
                    reward, details = compute_reward_from_info(ref_infos[i], gen_info)
                    gen_render = _viewport_screenshot(page, html)
                    ssim_score = ssim_fn(ref_renders[i], gen_render, channel_axis=2, data_range=255) if ref_renders[i].shape == gen_render.shape else 0.0
                    diff_mask = np.any(np.abs(ref_renders[i].astype(int) - gen_render.astype(int)) > 25, axis=2)
                    diff_pct = diff_mask.sum() / diff_mask.size
                    turn_result.update({"reward": reward, "ssim": ssim_score, "diff_pct": diff_pct, "details": details})
                except Exception as e:
                    turn_result["error"] = str(e)

            log(f"  Example {i+1}/{len(samples)} Turn 1: reward={turn_result['reward']:.3f}  SSIM={turn_result['ssim']:.3f}  diff={turn_result['diff_pct']:.1%}")

            eval_states.append({
                "turns": [turn_result],
                "html": html, "content": content,
                "done": html is None or turn_result["reward"] > 0.9,
                "convo": [
                    {"role": "system", "content": SYSTEM_PROMPT_AGENT},
                    {"role": "user", "content": [
                        {"type": "image", "image": ref_pils[i]},
                        {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot."},
                    ]},
                ],
            })

        # Turns 2+: parallel across all active examples
        for turn in range(1, args.turns):
            active = [(i, s) for i, s in enumerate(eval_states) if not s["done"]]
            if not active:
                break
            log(f"\nTurn {turn+1}: {len(active)} active examples")

            # Phase A: fire analyze calls in parallel
            analyze_futures = []
            for i, s in active:
                gen_render = _viewport_screenshot(page, s["html"])
                gen_pil = Image.fromarray(gen_render)

                s["convo"].append({"role": "assistant", "content": s["content"]})
                s["convo"].append({"role": "user", "content": [
                    {"type": "text", "text": "Here is the target screenshot:"},
                    {"type": "image", "image": ref_pils[i]},
                    {"type": "text", "text": "Here is what your HTML currently renders as:"},
                    {"type": "image", "image": gen_pil},
                    {"type": "text", "text": (
                        "Compare the two images. List the specific visual differences — "
                        "wrong colors, missing elements, incorrect sizing, layout issues. Be concise."
                    )},
                ]})

                analyze_prompt = build_vlm_prompt(s["convo"])
                analyze_futures.append(sampling_client.sample(
                    prompt=analyze_prompt, num_samples=1, sampling_params=sampling_params,
                ))

            # Collect analyze, build fix prompts
            for (i, s), future in zip(active, analyze_futures):
                result = future.result()
                parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
                analysis = get_text_content(parsed_msg)
                s["turns"][-1]["analysis"] = analysis

                s["convo"].append({"role": "assistant", "content": analysis})
                s["convo"].append({"role": "user", "content": (
                    "Now fix ALL the issues you identified. "
                    "Output the complete corrected HTML in ```html ... ```."
                )})

            # Phase B: fire fix calls in parallel
            fix_futures = []
            for i, s in active:
                fix_prompt = build_vlm_prompt(s["convo"])
                fix_futures.append(sampling_client.sample(
                    prompt=fix_prompt, num_samples=1, sampling_params=sampling_params,
                ))

            # Collect fix results
            for (i, s), future in zip(active, fix_futures):
                result = future.result()
                parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
                content = get_text_content(parsed_msg)
                html = extract_html_from_response(content)

                turn_result = {"turn": turn + 1, "html": html, "reward": -1.0, "ssim": 0.0, "diff_pct": 1.0}

                if html is not None:
                    try:
                        gen_info = extract_gen_info(page, html, size=IMG_SIZE)
                        reward, details = compute_reward_from_info(ref_infos[i], gen_info)
                        gen_render = _viewport_screenshot(page, html)
                        ssim_score = ssim_fn(ref_renders[i], gen_render, channel_axis=2, data_range=255) if ref_renders[i].shape == gen_render.shape else 0.0
                        diff_mask = np.any(np.abs(ref_renders[i].astype(int) - gen_render.astype(int)) > 25, axis=2)
                        diff_pct = diff_mask.sum() / diff_mask.size
                        turn_result.update({"reward": reward, "ssim": ssim_score, "diff_pct": diff_pct, "details": details})
                    except Exception as e:
                        turn_result["error"] = str(e)

                s["turns"].append(turn_result)
                s["html"] = html
                s["content"] = content
                s["done"] = html is None or turn_result["reward"] > 0.9

                log(f"  Example {i+1}/{len(samples)} Turn {turn+1}: reward={turn_result['reward']:.3f}  SSIM={turn_result['ssim']:.3f}  diff={turn_result['diff_pct']:.1%}")

        # Save all outputs
        for i in range(len(samples)):
            ex_dir = ex_dirs[i]
            turns = eval_states[i]["turns"]

            for t in turns:
                turn_num = t["turn"]
                if t.get("html"):
                    with open(os.path.join(ex_dir, f"turn{turn_num}.html"), "w") as f:
                        f.write(t["html"])
                    render_html_to_file(page, t["html"], os.path.join(ex_dir, f"turn{turn_num}.png"), full_page=False)

            final_turn = turns[-1] if turns else {"reward": -1.0}
            with open(os.path.join(ex_dir, "meta.json"), "w") as f:
                json.dump({"turns": turns, "final_reward": final_turn["reward"]}, f, indent=2, default=str)

            all_results.append({
                "example": i,
                "n_turns": len(turns),
                "rewards": [t["reward"] for t in turns],
                "ssims": [t["ssim"] for t in turns],
                "final_reward": final_turn["reward"],
            })

    else:
        # ── Sequential OpenAI eval (unchanged) ───────────────────────────
        for i, item in enumerate(samples):
            ex_dir = os.path.join(eval_dir, f"example_{i:02d}")
            os.makedirs(ex_dir, exist_ok=True)

            ref_html = item.get("reference_html") or item["html"]
            if item.get("screenshot") and os.path.exists(item["screenshot"]):
                ref_pil_orig = Image.open(item["screenshot"]).convert("RGB")
                ref_render = np.array(ref_pil_orig.resize((VIEWPORT_W, VIEWPORT_H)))
                ref_pil = Image.fromarray(ref_render)
                ref_info = extract_ref_info(page, ref_html, size=IMG_SIZE)
                ref_info["image"] = np.array(ref_pil_orig.resize((IMG_SIZE, IMG_SIZE)))
            else:
                ref_info = extract_ref_info(page, ref_html, size=IMG_SIZE)
                render_html(page, ref_html)
                ref_render = np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
                ref_pil = Image.fromarray(ref_render)

            ref_pil.save(os.path.join(ex_dir, "ref-render.png"))
            log(f"\nExample {i+1}/{args.n} ({len(ref_html)} chars)")

            turns = run_openai_agent(client, args.openai_model, ref_pil, ref_render, ref_info, page, args.turns)

            for t in turns:
                turn_num = t["turn"]
                if t["html"]:
                    with open(os.path.join(ex_dir, f"turn{turn_num}.html"), "w") as f:
                        f.write(t["html"])
                    render_html_to_file(page, t["html"], os.path.join(ex_dir, f"turn{turn_num}.png"), full_page=False)
                log(f"  Turn {turn_num}: reward={t['reward']:.3f}  SSIM={t['ssim']:.3f}  diff={t['diff_pct']:.1%}")

            final_turn = turns[-1] if turns else {"reward": -1.0}
            with open(os.path.join(ex_dir, "meta.json"), "w") as f:
                json.dump({"turns": turns, "final_reward": final_turn["reward"]}, f, indent=2, default=str)

            all_results.append({
                "example": i,
                "n_turns": len(turns),
                "rewards": [t["reward"] for t in turns],
                "ssims": [t["ssim"] for t in turns],
                "final_reward": final_turn["reward"],
            })

    # Summary
    log(f"\n{'='*60}")
    log(f"AGENT EVAL: {args.provider} ({args.openai_model if args.provider == 'openai' else MODEL})")
    log(f"{'='*60}")
    final_rewards = [r["final_reward"] for r in all_results]
    log(f"  Avg final reward: {np.mean(final_rewards):.3f}")
    log(f"  Avg SSIM (final): {np.mean([r['ssims'][-1] for r in all_results if r['ssims']]):.3f}")

    # Show per-turn improvement
    for turn_idx in range(args.turns):
        turn_rewards = [r["rewards"][turn_idx] for r in all_results if len(r["rewards"]) > turn_idx]
        if turn_rewards:
            log(f"  Turn {turn_idx+1} avg reward: {np.mean(turn_rewards):.3f} (n={len(turn_rewards)})")

    summary = {
        "provider": args.provider,
        "model": args.openai_model if args.provider == "openai" else MODEL,
        "n_examples": args.n,
        "max_turns": args.turns,
        "avg_final_reward": round(float(np.mean(final_rewards)), 4),
        "results": all_results,
    }
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nSaved to {eval_dir}/")
    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
