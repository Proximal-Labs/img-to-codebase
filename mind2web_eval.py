"""
Mind2Web interactive eval: takes action sequences + screenshots from Mind2Web,
has an agent generate HTML, then runs the action sequence via Playwright
and compares SSIM at each step.

The agent gets:
1. Initial screenshot → generates HTML (with analyze-fix turns)
2. We run the action sequence on the generated HTML
3. At each action step, we compare screenshots (gen vs ref)
4. Return per-step and overall SSIM

Usage:
    python mind2web_eval.py --n 3 --turns 2 --provider openai --openai_model gpt-5.4-2026-03-05
    python mind2web_eval.py --n 3 --turns 2 --provider tinker
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

from config import VIEWPORT_W, VIEWPORT_H, MODEL, LORA_RANK, RENDERER_NAME, IMG_SIZE
from reward import (
    render_html, extract_html_from_response, make_diff_image,
    extract_ref_info, extract_gen_info, compute_reward_from_info,
)
from train_agent import SYSTEM_PROMPT_AGENT, make_feedback_prompt

VIEWPORT = {"width": 1280, "height": 720}  # Mind2Web uses 1280 wide


def log(msg):
    print(msg, flush=True)


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def take_screenshot(page) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def compute_ssim(ref_img: np.ndarray, gen_img: np.ndarray) -> float:
    if ref_img.shape != gen_img.shape:
        gen_pil = Image.fromarray(gen_img).resize((ref_img.shape[1], ref_img.shape[0]))
        gen_img = np.array(gen_pil)
    return float(ssim_fn(ref_img, gen_img, channel_axis=2, data_range=255))


def load_mind2web_tasks(n: int, seed: int = 42) -> list[dict]:
    """Load and group Mind2Web actions by task (annotation_id)."""
    from datasets import load_dataset

    log("Loading Mind2Web dataset (streaming)...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train", streaming=True)

    # Group actions by annotation_id
    tasks = {}
    for i, row in enumerate(ds):
        if len(tasks) >= n * 5 and all(len(v["actions"]) >= 2 for v in tasks.values()):
            break
        if i > 5000:
            break

        ann_id = row["annotation_id"]
        if ann_id not in tasks:
            tasks[ann_id] = {
                "annotation_id": ann_id,
                "task": row["confirmed_task"],
                "website": row["website"],
                "domain": row["domain"],
                "actions": [],
            }

        # Extract action info
        op = json.loads(row["operation"]) if isinstance(row["operation"], str) else row["operation"]
        pos = json.loads(row["pos_candidates"][0]) if row["pos_candidates"] else {}
        if isinstance(pos, str):
            pos = json.loads(pos)

        bbox = None
        selector = None
        if isinstance(pos, dict):
            attrs = pos.get("attributes", "{}")
            if isinstance(attrs, str):
                attrs = json.loads(attrs)
            bbox = attrs.get("bounding_box_rect")
            el_id = attrs.get("id")
            if el_id:
                selector = f"#{el_id}"

        # Crop screenshot to viewport (Mind2Web has full-page screenshots)
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

    # Filter to tasks with 2+ actions and screenshots
    valid = [t for t in tasks.values() if len(t["actions"]) >= 2 and t["actions"][0]["screenshot"]]
    random.seed(seed)
    random.shuffle(valid)
    selected = valid[:n]

    log(f"  Selected {len(selected)} tasks (from {len(tasks)} total)")
    for t in selected:
        log(f"    [{t['website']}] {t['task'][:60]}... ({len(t['actions'])} actions)")

    return selected


MAX_FLOW_IMAGES = 5  # Max screenshots to show from the action flow


def build_flow_prompt_content(actions: list[dict], provider: str) -> list:
    """Build interleaved screenshot + action description content."""
    content = []
    content.append({"type": "text", "text": (
        "Here is a website user flow. Generate HTML/CSS/JS that reproduces "
        "this page with all the interactions working.\n"
    )})

    # Select up to MAX_FLOW_IMAGES steps (always include first, evenly sample rest)
    steps_with_screenshots = [(i, a) for i, a in enumerate(actions) if a.get("screenshot")]
    if len(steps_with_screenshots) > MAX_FLOW_IMAGES:
        # Always include first and last, sample middle
        selected = [steps_with_screenshots[0]]
        middle = steps_with_screenshots[1:-1]
        step_size = max(1, len(middle) // (MAX_FLOW_IMAGES - 2))
        selected.extend(middle[::step_size][:MAX_FLOW_IMAGES - 2])
        selected.append(steps_with_screenshots[-1])
    else:
        selected = steps_with_screenshots

    for j, (idx, action) in enumerate(selected):
        # Action description
        op = action["op"]
        desc = action["repr"]
        if j == 0:
            step_text = f"\nStep {j+1} — Initial page load:"
        else:
            step_text = f"\nStep {j+1} — {desc} ({op}):"

        content.append({"type": "text", "text": step_text})

        # Screenshot
        screenshot = action["screenshot"]
        if provider == "openai":
            img_b64 = pil_to_base64(screenshot.resize((1280, 720)))
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        else:
            content.append({"type": "image", "image": screenshot.resize((1280, 720))})

    content.append({"type": "text", "text": (
        "\n\nGenerate complete HTML/CSS that reproduces this page. "
        "Include JavaScript for the interactions shown above (clicks, navigation, dropdowns). "
        "Wrap your code in ```html ... ```."
    )})

    return content


def run_agent_generate(
    provider, client_or_sampler, renderer, ref_pil, page, max_turns,
    sampling_params=None, openai_model=None, actions=None,
) -> str | None:
    """Run multi-turn agent to generate HTML from screenshots + action flow."""
    ref_b64 = pil_to_base64(ref_pil) if provider == "openai" else None

    # Build initial prompt with flow screenshots
    if actions and len([a for a in actions if a.get("screenshot")]) > 1:
        user_content = build_flow_prompt_content(actions, provider)
    else:
        # Fallback: single screenshot
        if provider == "openai":
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot. The page should be interactive — buttons and links should be clickable elements."},
            ]
        else:
            user_content = [
                {"type": "image", "image": ref_pil},
                {"type": "text", "text": "Generate the HTML/CSS that reproduces this screenshot. The page should be interactive — buttons and links should be clickable elements."},
            ]

    if provider == "openai":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_AGENT},
            {"role": "user", "content": user_content},
        ]

        current_html = None
        for turn in range(max_turns):
            response = client_or_sampler.chat.completions.create(
                model=openai_model, messages=messages, max_completion_tokens=16384, temperature=0.3,
            )
            content = response.choices[0].message.content
            current_html = extract_html_from_response(content)

            if current_html is None or turn == max_turns - 1:
                break

            # Render and diff
            render_html(page, current_html)
            gen_img = take_screenshot(page)
            ref_arr = np.array(ref_pil)
            if ref_arr.shape != gen_img.shape:
                ref_arr = np.array(ref_pil.resize((gen_img.shape[1], gen_img.shape[0])))

            ssim_score = compute_ssim(ref_arr, gen_img)
            if ssim_score > 0.9:
                break

            diff_img = make_diff_image(ref_arr, gen_img, threshold=25)
            diff_b64 = pil_to_base64(Image.fromarray(diff_img))
            diff_mask = np.any(np.abs(ref_arr.astype(int) - gen_img.astype(int)) > 25, axis=2)
            diff_pct = diff_mask.sum() / diff_mask.size

            # Analyze
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{diff_b64}"}},
                {"type": "text", "text": (
                    f"Visual similarity: {ssim_score:.0%} ({diff_pct:.0%} pixels differ).\n"
                    f"List the specific visual differences in the red areas. Be concise."
                )},
            ]})
            analysis_resp = client_or_sampler.chat.completions.create(
                model=openai_model, messages=messages, max_completion_tokens=1024, temperature=0.3,
            )
            analysis = analysis_resp.choices[0].message.content
            messages.append({"role": "assistant", "content": analysis})
            messages.append({"role": "user", "content": "Fix ALL issues. Output complete corrected HTML in ```html ... ```."})

        return current_html

    else:
        # Tinker path
        from tinker_cookbook.renderers import get_text_content
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT_AGENT},
            {"role": "user", "content": user_content},
        ]

        current_html = None
        for turn in range(max_turns):
            prompt = renderer.build_generation_prompt(convo)
            result = client_or_sampler.sample(prompt=prompt, num_samples=1, sampling_params=sampling_params).result()
            parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
            content = get_text_content(parsed_msg)
            current_html = extract_html_from_response(content)

            if current_html is None or turn == max_turns - 1:
                break

            render_html(page, current_html)
            gen_img = take_screenshot(page)
            ref_arr = np.array(ref_pil)
            if ref_arr.shape != gen_img.shape:
                ref_arr = np.array(ref_pil.resize((gen_img.shape[1], gen_img.shape[0])))

            ssim_score = compute_ssim(ref_arr, gen_img)
            if ssim_score > 0.9:
                break

            diff_img = make_diff_image(ref_arr, gen_img, threshold=25)
            diff_pil = Image.fromarray(diff_img)
            diff_mask = np.any(np.abs(ref_arr.astype(int) - gen_img.astype(int)) > 25, axis=2)
            diff_pct = diff_mask.sum() / diff_mask.size

            convo.append({"role": "assistant", "content": content})
            convo.append({"role": "user", "content": [
                {"type": "image", "image": ref_pil},
                {"type": "image", "image": diff_pil},
                {"type": "text", "text": (
                    f"Visual similarity: {ssim_score:.0%} ({diff_pct:.0%} pixels differ).\n"
                    f"List the specific visual differences. Be concise."
                )},
            ]})
            analyze_prompt = renderer.build_generation_prompt(convo)
            analyze_result = client_or_sampler.sample(prompt=analyze_prompt, num_samples=1, sampling_params=sampling_params).result()
            analyze_msg, _ = renderer.parse_response(analyze_result.sequences[0].tokens)
            analysis = get_text_content(analyze_msg)
            convo.append({"role": "assistant", "content": analysis})
            convo.append({"role": "user", "content": "Fix ALL issues. Output complete corrected HTML in ```html ... ```."})

        return current_html


def run_action_sequence(page, actions: list[dict], out_dir: str) -> list[dict]:
    """Run Mind2Web actions on the page and capture screenshots."""
    results = []

    for i, action in enumerate(actions):
        ref_screenshot = action.get("screenshot")
        if ref_screenshot is None:
            continue

        # Capture current state
        gen_img = take_screenshot(page)
        ref_arr = np.array(ref_screenshot.resize((gen_img.shape[1], gen_img.shape[0])))

        ssim_score = compute_ssim(ref_arr, gen_img)

        # Save
        Image.fromarray(gen_img).save(os.path.join(out_dir, f"step_{i}_gen.png"))
        Image.fromarray(ref_arr).save(os.path.join(out_dir, f"step_{i}_ref.png"))
        diff_img = make_diff_image(ref_arr, gen_img, threshold=25)
        Image.fromarray(diff_img).save(os.path.join(out_dir, f"step_{i}_diff.png"))

        results.append({
            "step": i,
            "action": action["repr"],
            "op": action["op"],
            "ssim": ssim_score,
        })

        # Execute action
        op = action["op"]
        selector = action.get("selector")
        bbox = action.get("bbox")

        if op == "CLICK" and selector:
            try:
                page.click(selector, timeout=3000)
                page.wait_for_timeout(500)
            except Exception:
                # Try clicking by coordinates from bbox
                if bbox:
                    coords = [float(x) for x in bbox.split(",")]
                    if len(coords) >= 4:
                        x, y = coords[0] + coords[2] / 2, coords[1] + coords[3] / 2
                        page.mouse.click(x, y)
                        page.wait_for_timeout(500)

        elif op == "TYPE" and selector:
            try:
                page.fill(selector, action.get("value", ""), timeout=3000)
                page.wait_for_timeout(300)
            except Exception:
                pass

        elif op == "SELECT" and selector:
            try:
                page.select_option(selector, action.get("value", ""), timeout=3000)
                page.wait_for_timeout(300)
            except Exception:
                pass

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="Number of tasks")
    parser.add_argument("--turns", type=int, default=2, help="Agent turns for HTML generation")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "tinker"])
    parser.add_argument("--openai_model", type=str, default="gpt-5.4-2026-03-05")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datetime import datetime
    eval_name = args.name or f"mind2web_{args.provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join("eval_output", eval_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load tasks
    tasks = load_mind2web_tasks(args.n, seed=args.seed)

    # Setup browser
    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    page = browser.new_page(viewport=VIEWPORT)

    # Setup provider
    client_or_sampler = None
    renderer = None
    sampling_params = None
    openai_model = None

    if args.provider == "openai":
        from openai import OpenAI
        client_or_sampler = OpenAI()
        openai_model = args.openai_model
    else:
        import tinker
        from tinker import types
        from transformers import AutoImageProcessor
        from tinker_cookbook import renderers as rnd
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        tokenizer = get_tokenizer(MODEL)
        image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
        renderer = rnd.get_renderer(RENDERER_NAME, tokenizer, image_processor=image_processor)
        sampling_params = types.SamplingParams(max_tokens=4096, stop=renderer.get_stop_sequences(), temperature=0.3)

        sc = tinker.ServiceClient()
        if args.model_path:
            client_or_sampler = sc.create_sampling_client(model_path=args.model_path)
        else:
            tc = sc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
            client_or_sampler = tc.save_weights_and_get_sampling_client()

    all_results = []

    for task_idx, task in enumerate(tasks):
        task_dir = os.path.join(out_dir, f"task_{task_idx:02d}")
        os.makedirs(task_dir, exist_ok=True)

        log(f"\n{'='*60}")
        log(f"Task {task_idx+1}/{len(tasks)}: [{task['website']}] {task['task'][:80]}")
        log(f"  Actions: {len(task['actions'])}")

        # Get initial screenshot as model input
        initial_screenshot = task["actions"][0]["screenshot"]
        ref_pil = initial_screenshot.resize((VIEWPORT["width"], VIEWPORT["height"]))
        ref_pil.save(os.path.join(task_dir, "reference.png"))

        # Agent generates HTML
        log("  Generating HTML...")
        gen_html = run_agent_generate(
            args.provider, client_or_sampler, renderer, ref_pil, page,
            args.turns, sampling_params, openai_model, actions=task["actions"],
        )

        if gen_html is None:
            log("  Failed to generate HTML")
            all_results.append({"task": task["task"], "website": task["website"], "ssim_initial": 0.0, "ssim_avg_steps": 0.0, "steps": []})
            continue

        with open(os.path.join(task_dir, "generated.html"), "w") as f:
            f.write(gen_html)

        # Render generated HTML
        render_html(page, gen_html)
        page.wait_for_timeout(500)

        # Initial SSIM (before any actions)
        gen_img = take_screenshot(page)
        ref_arr = np.array(ref_pil.resize((gen_img.shape[1], gen_img.shape[0])))
        initial_ssim = compute_ssim(ref_arr, gen_img)
        log(f"  Initial SSIM: {initial_ssim:.3f}")

        # Run action sequence
        log("  Running action sequence...")
        step_results = run_action_sequence(page, task["actions"], task_dir)

        for r in step_results:
            log(f"    Step {r['step']} [{r['op']}]: SSIM={r['ssim']:.3f}  ({r['action'][:50]})")

        step_ssims = [r["ssim"] for r in step_results]
        task_result = {
            "task": task["task"],
            "website": task["website"],
            "n_actions": len(task["actions"]),
            "ssim_initial": round(initial_ssim, 4),
            "ssim_avg_steps": round(float(np.mean(step_ssims)), 4) if step_ssims else 0.0,
            "steps": step_results,
        }
        all_results.append(task_result)

        with open(os.path.join(task_dir, "results.json"), "w") as f:
            json.dump(task_result, f, indent=2, default=str)

    # Summary
    log(f"\n{'='*60}")
    log("MIND2WEB EVAL SUMMARY")
    log(f"{'='*60}")
    initial_ssims = [r["ssim_initial"] for r in all_results]
    step_ssims = [r["ssim_avg_steps"] for r in all_results if r["ssim_avg_steps"] > 0]
    log(f"  Avg initial SSIM: {np.mean(initial_ssims):.3f}")
    log(f"  Avg step SSIM:    {np.mean(step_ssims):.3f}" if step_ssims else "  No step results")

    summary = {
        "provider": args.provider,
        "model": openai_model or MODEL,
        "n_tasks": len(tasks),
        "max_turns": args.turns,
        "avg_initial_ssim": round(float(np.mean(initial_ssims)), 4),
        "avg_step_ssim": round(float(np.mean(step_ssims)), 4) if step_ssims else 0.0,
        "results": all_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\nSaved to {out_dir}/")

    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
