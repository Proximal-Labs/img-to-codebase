"""
Multi-page eval: given screenshots of different pages/states of a website,
generate HTML that contains all those pages.

Instead of testing specific click sequences, we:
1. Show model N page screenshots ("Home", "Marketplace", "Cart")
2. Model generates multi-page HTML with routing
3. We discover all pages/states in the generated HTML
4. Match each generated page to closest reference by SSIM
5. Score: how many pages exist, how well do they match

Usage:
    python multipage_eval.py --n 3 --provider openai --openai_model gpt-5.4-2026-03-05
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

from config import MODEL, LORA_RANK, RENDERER_NAME
from reward import render_html, extract_html_from_response, make_diff_image
from train_agent import SYSTEM_PROMPT_AGENT

VIEWPORT = {"width": 1280, "height": 720}


def log(msg):
    print(msg, flush=True)


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def take_screenshot(page) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0])))
    return float(ssim_fn(a, b, channel_axis=2, data_range=255))


def load_multipage_tasks(n: int, seed: int = 42) -> list[dict]:
    """Load Mind2Web tasks and group screenshots by task as 'pages'."""
    from datasets import load_dataset

    log("Loading Mind2Web (streaming)...")
    ds = load_dataset("osunlp/Multimodal-Mind2Web", split="train", streaming=True)

    tasks = {}
    for i, row in enumerate(ds):
        if len(tasks) >= n * 5:
            break
        if i > 5000:
            break

        ann_id = row["annotation_id"]
        if ann_id not in tasks:
            tasks[ann_id] = {
                "annotation_id": ann_id,
                "task": row["confirmed_task"],
                "website": row["website"],
                "pages": [],
                "seen_screenshots": set(),
            }

        screenshot = row["screenshot"]
        if screenshot is None:
            continue

        # Crop to viewport
        page_img = screenshot.crop((0, 0, min(screenshot.width, 1280), min(screenshot.height, 720)))

        # Deduplicate similar screenshots (skip if too similar to existing)
        page_arr = np.array(page_img.resize((320, 180)))
        is_duplicate = False
        for existing_arr in tasks[ann_id]["seen_screenshots"]:
            if np.abs(page_arr.astype(int) - existing_arr.astype(int)).mean() < 5:
                is_duplicate = True
                break

        if not is_duplicate:
            action_repr = row["target_action_reprs"]
            tasks[ann_id]["pages"].append({
                "image": page_img,
                "description": action_repr,
            })
            tasks[ann_id]["seen_screenshots"].add(tuple(page_arr.flatten()[:1000]))

    # Filter tasks with 3+ distinct pages
    valid = [t for t in tasks.values() if len(t["pages"]) >= 3]
    # Clean up non-serializable field
    for t in valid:
        del t["seen_screenshots"]

    random.seed(seed)
    random.shuffle(valid)
    selected = valid[:n]

    log(f"  Selected {len(selected)} tasks with 3+ pages")
    for t in selected:
        log(f"    [{t['website']}] {t['task'][:60]}... ({len(t['pages'])} pages)")

    return selected


def discover_pages(page) -> list[dict]:
    """Find all pages/states in the generated HTML by looking for
    hidden divs, route handlers, nav links, etc."""
    pages_found = page.evaluate("""() => {
        const pages = [];

        // Method 1: Find divs with display:none that look like pages
        const allDivs = document.querySelectorAll('div[id], section[id], main[id]');
        for (const div of allDivs) {
            const style = getComputedStyle(div);
            const id = div.id;
            // Skip tiny elements
            if (div.offsetWidth < 100) continue;
            pages.push({
                id: id,
                display: style.display,
                visible: style.display !== 'none' && style.visibility !== 'hidden',
                height: div.offsetHeight,
                tag: div.tagName.toLowerCase(),
            });
        }

        return pages;
    }""")
    return pages_found


def screenshot_each_page(page, pages_found: list[dict]) -> list[dict]:
    """Show each hidden page one at a time and screenshot it."""
    results = []

    for p in pages_found:
        pid = p["id"]
        if not pid:
            continue

        # Show this page, hide others
        page.evaluate(f"""(targetId) => {{
            const allPages = document.querySelectorAll('div[id], section[id], main[id]');
            for (const el of allPages) {{
                if (el.offsetWidth < 100) continue;
                const isPage = el.id === targetId;
                el.style.display = isPage ? 'block' : 'none';
            }}
        }}""", pid)
        page.wait_for_timeout(300)

        img = take_screenshot(page)
        # Check if it has meaningful content (not blank)
        nonwhite = np.any(np.array(img) < 240, axis=2).sum() / img[:, :, 0].size
        if nonwhite < 0.02:
            continue

        results.append({
            "id": pid,
            "image": img,
            "display": p["display"],
            "was_visible": p["visible"],
        })

    return results


def match_pages(ref_pages: list[dict], gen_pages: list[dict]) -> list[dict]:
    """Match generated pages to reference pages by best SSIM."""
    matches = []
    used_gen = set()

    for ref in ref_pages:
        ref_img = np.array(ref["image"].resize((1280, 720)))
        best_ssim = 0.0
        best_idx = -1

        for j, gen in enumerate(gen_pages):
            if j in used_gen:
                continue
            ssim = compute_ssim(ref_img, gen["image"])
            if ssim > best_ssim:
                best_ssim = ssim
                best_idx = j

        match = {
            "ref_description": ref.get("description", ""),
            "ssim": best_ssim,
            "matched": best_idx >= 0 and best_ssim > 0.1,
        }
        if best_idx >= 0:
            match["gen_page_id"] = gen_pages[best_idx]["id"]
            used_gen.add(best_idx)

        matches.append(match)

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--openai_model", type=str, default="gpt-5.4-2026-03-05")
    parser.add_argument("--anthropic_model", type=str, default="claude-opus-4-6")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datetime import datetime
    eval_name = args.name or f"multipage_{args.provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join("eval_output", eval_name)
    os.makedirs(out_dir, exist_ok=True)

    tasks = load_multipage_tasks(args.n, seed=args.seed)

    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    page = browser.new_page(viewport=VIEWPORT)

    # Setup provider
    if args.provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model_name = args.openai_model
    else:
        import anthropic
        client = anthropic.Anthropic()
        model_name = args.anthropic_model

    all_results = []

    for task_idx, task in enumerate(tasks):
        task_dir = os.path.join(out_dir, f"task_{task_idx:02d}")
        os.makedirs(task_dir, exist_ok=True)

        log(f"\n{'='*60}")
        log(f"Task {task_idx+1}/{len(tasks)}: [{task['website']}] {task['task'][:80]}")
        log(f"  {len(task['pages'])} distinct pages")

        # Build prompt with all page screenshots
        content = [{"type": "text", "text": (
            "This website has multiple pages/states. Generate a single HTML file with "
            "all pages built in. Use JavaScript to handle navigation between pages "
            "(show/hide divs, or similar routing). Each page should be visually accurate.\n\n"
            "Here are the pages:\n"
        )}]

        for j, pg in enumerate(task["pages"][:6]):  # Max 6 pages
            pg_img = pg["image"].resize((1280, 720))
            pg_img.save(os.path.join(task_dir, f"ref_page_{j}.png"))

            desc = pg.get("description", f"Page {j+1}")
            content.append({"type": "text", "text": f"\nPage {j+1} — {desc}:"})

            if args.provider == "openai":
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pil_to_base64(pg_img)}"}})
            else:
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": pil_to_base64(pg_img)}})

        content.append({"type": "text", "text": (
            "\n\nGenerate complete HTML/CSS/JS with all pages. "
            "Wrap in ```html ... ```."
        )})

        # Generate
        log("  Generating...")
        if args.provider == "openai":
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You generate multi-page HTML websites. Include all pages in a single file with JavaScript routing."},
                    {"role": "user", "content": content},
                ],
                max_completion_tokens=32768, temperature=0.3,
            )
            html = extract_html_from_response(resp.choices[0].message.content)
        else:
            text = ""
            with client.messages.stream(
                model=model_name,
                system="You generate multi-page HTML websites. Include all pages in a single file with JavaScript routing.",
                messages=[{"role": "user", "content": content}],
                max_tokens=32768, temperature=0.3,
            ) as stream:
                for chunk in stream.text_stream:
                    text += chunk
            html = extract_html_from_response(text)

        if html is None:
            log("  Failed to generate HTML")
            all_results.append({"task": task["task"], "pages_found": 0, "pages_matched": 0, "avg_ssim": 0.0})
            continue

        with open(os.path.join(task_dir, "generated.html"), "w") as f:
            f.write(html)
        log(f"  Generated {len(html)} chars")

        # Render and discover pages
        render_html(page, html)
        page.wait_for_timeout(500)

        # Screenshot initial state
        initial_img = take_screenshot(page)
        Image.fromarray(initial_img).save(os.path.join(task_dir, "initial.png"))

        pages_found = discover_pages(page)
        log(f"  Found {len(pages_found)} page elements ({sum(1 for p in pages_found if not p['visible'])} hidden)")

        # Screenshot each page
        gen_pages = screenshot_each_page(page, pages_found)
        log(f"  Screenshotted {len(gen_pages)} pages with content")

        for j, gp in enumerate(gen_pages):
            Image.fromarray(gp["image"]).save(os.path.join(task_dir, f"gen_page_{j}_{gp['id']}.png"))

        # Match to reference pages
        matches = match_pages(task["pages"][:6], gen_pages)
        matched_count = sum(1 for m in matches if m["matched"])
        ssims = [m["ssim"] for m in matches]
        avg_ssim = float(np.mean(ssims)) if ssims else 0.0

        log(f"  Matched {matched_count}/{len(task['pages'][:6])} pages, avg SSIM: {avg_ssim:.3f}")
        for m in matches:
            status = "✓" if m["matched"] else "✗"
            log(f"    {status} {m['ref_description'][:50]}: SSIM={m['ssim']:.3f}" +
                (f" → {m.get('gen_page_id', '?')}" if m["matched"] else ""))

        result = {
            "task": task["task"],
            "website": task["website"],
            "ref_pages": len(task["pages"][:6]),
            "pages_found": len(gen_pages),
            "pages_matched": matched_count,
            "avg_ssim": round(avg_ssim, 4),
            "matches": matches,
        }
        all_results.append(result)

        with open(os.path.join(task_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Summary
    log(f"\n{'='*60}")
    log("MULTI-PAGE EVAL SUMMARY")
    log(f"{'='*60}")
    total_ref = sum(r.get("ref_pages", 0) for r in all_results)
    total_matched = sum(r.get("pages_matched", 0) for r in all_results)
    avg_ssims = [r["avg_ssim"] for r in all_results if r["avg_ssim"] > 0]
    log(f"  Pages matched: {total_matched}/{total_ref}")
    log(f"  Avg SSIM: {np.mean(avg_ssims):.3f}" if avg_ssims else "  No matches")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"results": all_results, "total_matched": total_matched, "total_ref": total_ref}, f, indent=2, default=str)

    log(f"\nSaved to {out_dir}/")
    browser.close()
    pw.stop()


if __name__ == "__main__":
    main()
