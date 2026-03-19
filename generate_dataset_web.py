"""
Download and prepare website HTML/screenshot pairs from HuggingFaceM4/WebSight v0.2.

Filters for HTML that:
  - Is short enough to fit in ~1024 tokens (≤ MAX_HTML_CHARS)
  - Renders meaningful visual content (not blank/broken)

Outputs manifest.json with: id, screenshot, html (snippet), reference_html (full page).
"""

import json
import io
import os
import re
import random

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from config import (
    DATA_DIR, VIEWPORT_W, VIEWPORT_H,
    WEBSIGHT_TARGET, WEBSIGHT_CANDIDATES, MAX_HTML_CHARS,
)


SCREENSHOTS_DIR = os.path.join(DATA_DIR, "screenshots")
MIN_CONTENT_RATIO = 0.02


def is_full_html(html: str) -> bool:
    stripped = html.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")


def extract_body_content(html: str) -> str | None:
    match = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_inline_styles(html: str) -> str:
    styles = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    return "<style>" + "\n".join(styles) + "</style>" if styles else ""


def has_meaningful_content(img_arr: np.ndarray) -> bool:
    diff = np.any(img_arr != 255, axis=2)
    return diff.sum() / diff.size > MIN_CONTENT_RATIO


def render_html(page, html: str) -> np.ndarray | None:
    try:
        if is_full_html(html):
            page.set_content(html)
        else:
            page.set_content(
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<style>body{margin:20px;background:#fff;}</style>"
                f"</head><body>{html}</body></html>"
            )
        page.wait_for_timeout(300)
        screenshot_bytes = page.screenshot()
        return np.array(Image.open(io.BytesIO(screenshot_bytes)).convert("RGB"))
    except Exception:
        return None


def fetch_websight_samples(n_target: int, n_candidates: int) -> list[dict]:
    """Download and filter samples from WebSight v0.2."""
    from datasets import load_dataset

    print(f"Loading WebSight v0.2 ({n_candidates} candidates)...")
    ds = load_dataset(
        "HuggingFaceM4/WebSight",
        name="v0.2",
        split=f"train[:{n_candidates}]",
    )

    samples = []
    skipped = 0

    for i, row in enumerate(ds):
        html = row["text"]
        if len(html) > MAX_HTML_CHARS:
            skipped += 1
            continue

        if is_full_html(html):
            body = extract_body_content(html)
            styles = extract_inline_styles(html)
            if body is None:
                continue
            snippet = (styles + "\n" + body).strip() if styles else body
            if len(snippet) > MAX_HTML_CHARS:
                skipped += 1
                continue
        else:
            snippet = html

        samples.append({
            "html": snippet,
            "reference_html": html,  # full HTML for DOM-based reward
            "source_idx": i,
        })

    print(f"  Passed filter: {len(samples)} (skipped {skipped} too long)")

    if len(samples) > n_target:
        samples = random.sample(samples, n_target)
        print(f"  Sampled down to {n_target}")

    return samples


def render_and_save(samples: list[dict]) -> list[dict]:
    """Render all samples to screenshots, filtering out broken/blank ones."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    manifest = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

        for sample in samples:
            render_source = sample.get("reference_html", sample["html"])
            img_arr = render_html(page, render_source)

            if img_arr is None or not has_meaningful_content(img_arr):
                continue

            idx = len(manifest)
            screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{idx:04d}.png")
            Image.fromarray(img_arr).save(screenshot_path)

            manifest.append({
                "id": idx,
                "screenshot": screenshot_path,
                "html": sample["html"],
                "reference_html": sample["reference_html"],
            })

            if len(manifest) % 100 == 0:
                print(f"  Rendered {len(manifest)} valid samples...")

        browser.close()

    return manifest


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    random.seed(42)

    samples = fetch_websight_samples(WEBSIGHT_TARGET, WEBSIGHT_CANDIDATES)
    random.shuffle(samples)
    print(f"\nTotal candidates: {len(samples)}")

    print("\nRendering and validating screenshots...")
    manifest = render_and_save(samples)

    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Saved {len(manifest)} pairs to {manifest_path}")

    lengths = [len(m["html"]) for m in manifest]
    if lengths:
        print(f"\nHTML length stats:")
        print(f"  min: {min(lengths)} chars")
        print(f"  max: {max(lengths)} chars")
        print(f"  mean: {sum(lengths) / len(lengths):.0f} chars")
        print(f"  median: {sorted(lengths)[len(lengths)//2]} chars")


if __name__ == "__main__":
    main()
