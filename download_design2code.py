"""
Download Design2Code benchmark dataset for evaluation.

Saves screenshots and HTML pairs in data/design2code/ with a manifest.
"""

import json
import io
import os

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

from config import DATA_DIR, VIEWPORT_W, VIEWPORT_H


DESIGN2CODE_DIR = os.path.join(DATA_DIR, "design2code")
SCREENSHOTS_DIR = os.path.join(DESIGN2CODE_DIR, "screenshots")


def main():
    from datasets import load_dataset

    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    print("Loading SALT-NLP/Design2Code-hf...")
    ds = load_dataset("SALT-NLP/Design2Code-hf", split="train")
    print(f"  {len(ds)} examples")

    manifest = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": VIEWPORT_W, "height": VIEWPORT_H})

        for i, row in enumerate(ds):
            html = row["text"]
            screenshot_path = os.path.join(SCREENSHOTS_DIR, f"{i:04d}.png")

            # Render the HTML to get a consistent screenshot at our viewport
            try:
                page.set_content(html)
                page.wait_for_timeout(300)
                page.screenshot(path=screenshot_path)
            except Exception:
                continue

            manifest.append({
                "id": i,
                "screenshot": screenshot_path,
                "html": html,
                "reference_html": html,
            })

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(ds)}")

        browser.close()

    manifest_path = os.path.join(DESIGN2CODE_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Saved {len(manifest)} Design2Code examples to {manifest_path}")


if __name__ == "__main__":
    main()
