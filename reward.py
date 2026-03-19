"""
Multi-signal reward function for screenshot-to-HTML RL.

Designed to be smoothly climbable — each signal degrades gracefully
rather than cliff-edging on mismatches.

Reward signals:
  0.25 - CLIP perceptual similarity (smooth anchor)
  0.25 - global text content match (all visible text compared)
  0.20 - layout / block position matching (meaningful elements only)
  0.15 - color palette similarity (dominant colors)
  0.15 - visual SSIM (pixel-level)
"""

import io
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import Page

from config import VIEWPORT_W, VIEWPORT_H


# ── HTML extraction ───────────────────────────────────────────────────────────

def extract_html_from_response(text: str) -> str | None:
    """Extract HTML from a model response."""
    match = re.search(r"```html\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*(<!?[^`]*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped = text.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return stripped

    match = re.search(
        r"(<(?:style|div|span|h[1-6]|p|section|header|nav|main|footer|html|!DOCTYPE)[^>]*>.*)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    return None


# ── HTML rendering ────────────────────────────────────────────────────────────

def is_full_html(html: str) -> bool:
    stripped = html.strip().lower()
    return stripped.startswith("<!doctype") or stripped.startswith("<html")


def _wrap_snippet(html_snippet: str) -> str:
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        "<style>body{margin:20px;background:#fff;}</style>"
        "</head><body>"
        f"{html_snippet}"
        "</body></html>"
    )


def render_html(page: Page, html_snippet: str):
    if is_full_html(html_snippet):
        page.set_content(html_snippet)
    else:
        page.set_content(_wrap_snippet(html_snippet))
    page.wait_for_timeout(100)


def render_html_to_image(page: Page, html_snippet: str, size: int = 256) -> np.ndarray:
    render_html(page, html_snippet)
    screenshot_bytes = page.screenshot()
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB").resize((size, size))
    return np.array(img)


def render_html_to_file(page: Page, html_snippet: str | None, save_path: str, full_page: bool = True) -> bool:
    if html_snippet is None:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False
    try:
        render_html(page, html_snippet)
        page.wait_for_timeout(100)
        page.screenshot(path=save_path, full_page=full_page)
        return True
    except Exception:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False


def load_reference_image(path: str, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size))
    return np.array(img)


# ── CLIP similarity ───────────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None


def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k",
        )
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
    return _clip_model, _clip_preprocess


def clip_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    model, preprocess = _get_clip()
    with torch.no_grad():
        tensor_a = preprocess(Image.fromarray(img_a)).unsqueeze(0)
        tensor_b = preprocess(Image.fromarray(img_b)).unsqueeze(0)
        feat_a = model.encode_image(tensor_a)
        feat_b = model.encode_image(tensor_b)
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
        sim = (feat_a @ feat_b.T).item()
    # CLIP cosine sim is usually 0.5-1.0 range; normalize to 0-1
    return float(max(0.0, (sim - 0.5) * 2.0))


# ── Global text comparison ───────────────────────────────────────────────────

def _extract_visible_text(page: Page) -> str:
    """Extract all visible text from the rendered page."""
    return page.evaluate("""() => {
        return document.body ? document.body.innerText.trim() : '';
    }""")


def text_similarity(ref_text: str, gen_text: str) -> float:
    """Global fuzzy text similarity. Smooth: partial matches score partial credit."""
    if not ref_text and not gen_text:
        return 1.0
    if not ref_text or not gen_text:
        return 0.0
    return SequenceMatcher(None, ref_text.lower(), gen_text.lower()).ratio()


# ── Layout / block matching ──────────────────────────────────────────────────

MEANINGFUL_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "a", "button", "input",
    "img", "nav", "header", "footer", "main", "section", "article",
    "li", "td", "th", "label", "span", "textarea", "select",
}


def _extract_meaningful_blocks(page: Page) -> list[dict]:
    """Extract only visually meaningful DOM elements (not wrapper divs)."""
    tags_list = list(MEANINGFUL_TAGS)
    blocks = page.evaluate("""(tagsList) => {
        const meaningfulTags = new Set(tagsList);
        const blocks = [];
        const els = document.querySelectorAll('*');
        for (const el of els) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 5 || rect.height < 5) continue;
            if (rect.width >= window.innerWidth * 0.99 && rect.height >= window.innerHeight * 0.99) continue;

            const tag = el.tagName.toLowerCase();
            const style = getComputedStyle(el);
            const hasBg = style.backgroundColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor !== 'transparent';
            const hasBorder = style.borderWidth && style.borderWidth !== '0px';

            let directText = '';
            for (const node of el.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    directText += node.textContent;
                }
            }
            directText = directText.trim();

            const isMeaningful = meaningfulTags.has(tag) || directText.length > 0 || hasBg || hasBorder;
            if (!isMeaningful) continue;

            blocks.push({
                tag: tag,
                x: rect.x,
                y: rect.y,
                w: rect.width,
                h: rect.height,
                text: directText.substring(0, 200),
            });
        }
        return blocks;
    }""", tags_list)
    return blocks


def _iou(a: dict, b: dict) -> float:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def layout_score(ref_blocks: list[dict], gen_blocks: list[dict]) -> float:
    """
    Smooth layout matching score.

    For each ref block, find best matching gen block by IoU.
    Score = average of best IoUs, with a soft penalty for count mismatch.
    No harsh match_ratio multiplier.
    """
    if not ref_blocks and not gen_blocks:
        return 1.0
    if not ref_blocks or not gen_blocks:
        return 0.0

    used_gen = set()
    ious = []

    for ref in ref_blocks:
        best_iou = 0.0
        best_idx = -1
        for j, gen in enumerate(gen_blocks):
            if j in used_gen:
                continue
            score = _iou(ref, gen)
            if score > best_iou:
                best_iou = score
                best_idx = j

        ious.append(best_iou)
        if best_idx >= 0 and best_iou > 0.05:
            used_gen.add(best_idx)

    avg_iou = sum(ious) / len(ious)

    # Soft penalty for count mismatch (not multiplicative)
    count_ratio = min(len(ref_blocks), len(gen_blocks)) / max(len(ref_blocks), len(gen_blocks))
    count_penalty = 0.5 + 0.5 * count_ratio  # ranges from 0.5 to 1.0

    return float(avg_iou * count_penalty)


# ── Color palette similarity ─────────────────────────────────────────────────

def _extract_colors(page: Page) -> list[tuple[int, int, int]]:
    """Extract background colors of visible elements."""
    colors = page.evaluate("""() => {
        const colors = [];
        const els = document.querySelectorAll('*');
        for (const el of els) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 10 || rect.height < 10) continue;
            const style = getComputedStyle(el);
            const bg = style.backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                const match = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                if (match) {
                    colors.push([parseInt(match[1]), parseInt(match[2]), parseInt(match[3])]);
                }
            }
            const fg = style.color;
            if (fg) {
                const match = fg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                if (match) {
                    colors.push([parseInt(match[1]), parseInt(match[2]), parseInt(match[3])]);
                }
            }
        }
        return colors;
    }""")
    return [tuple(c) for c in colors]


def _quantize_color(c: tuple[int, int, int], step: int = 32) -> tuple[int, int, int]:
    """Quantize color to reduce noise."""
    return (c[0] // step * step, c[1] // step * step, c[2] // step * step)


def color_palette_similarity(ref_colors: list, gen_colors: list) -> float:
    """Compare color palettes using quantized histogram overlap."""
    if not ref_colors and not gen_colors:
        return 1.0
    if not ref_colors or not gen_colors:
        return 0.0

    ref_hist = Counter(_quantize_color(c) for c in ref_colors)
    gen_hist = Counter(_quantize_color(c) for c in gen_colors)

    # Histogram intersection (normalized)
    all_colors = set(ref_hist.keys()) | set(gen_hist.keys())
    intersection = sum(min(ref_hist.get(c, 0), gen_hist.get(c, 0)) for c in all_colors)
    total = max(sum(ref_hist.values()), sum(gen_hist.values()))

    return float(intersection / total) if total > 0 else 0.0


# ── Visual similarity ────────────────────────────────────────────────────────

def _get_content_bbox(img_arr: np.ndarray, bg_color: int = 255, margin: int = 5):
    diff = np.any(img_arr != bg_color, axis=2)
    rows = np.any(diff, axis=1)
    cols = np.any(diff, axis=0)
    if not rows.any():
        h, w = img_arr.shape[:2]
        return 0, h, 0, w
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    h, w = img_arr.shape[:2]
    return max(0, y0 - margin), min(h, y1 + margin + 1), max(0, x0 - margin), min(w, x1 + margin + 1)


def visual_similarity(ref_img: np.ndarray, gen_img: np.ndarray) -> float:
    y0, y1, x0, x1 = _get_content_bbox(ref_img)
    ref_crop = ref_img[y0:y1, x0:x1]
    gen_crop = gen_img[y0:y1, x0:x1]

    mse = np.mean((ref_crop.astype(float) - gen_crop.astype(float)) ** 2) / (255.0 ** 2)
    pixel_score = 1.0 - mse

    if ref_crop.shape[0] >= 7 and ref_crop.shape[1] >= 7:
        ssim_score = ssim(ref_crop, gen_crop, channel_axis=2, data_range=255)
    else:
        ssim_score = pixel_score

    return float(0.3 * pixel_score + 0.7 * ssim_score)


# ── Combined reward ──────────────────────────────────────────────────────────

WEIGHTS = {
    "clip": 0.25,
    "text": 0.25,
    "layout": 0.20,
    "color": 0.15,
    "visual": 0.15,
}


def compute_reward(
    generated_html: str | None,
    reference_html: str,
    ref_image: np.ndarray,
    page: Page,
    size: int = 256,
) -> tuple[float, dict]:
    """
    Compute multi-signal reward. Each signal is smooth and independently climbable.

    Returns (reward, details) where reward is in [-1, 1].
    """
    if generated_html is None:
        return -1.0, {k: 0.0 for k in WEIGHTS}

    try:
        gen_image = render_html_to_image(page, generated_html, size=size)

        # Global text comparison
        render_html(page, reference_html)
        ref_text = _extract_visible_text(page)
        ref_blocks = _extract_meaningful_blocks(page)
        ref_colors = _extract_colors(page)

        render_html(page, generated_html)
        gen_text = _extract_visible_text(page)
        gen_blocks = _extract_meaningful_blocks(page)
        gen_colors = _extract_colors(page)

    except Exception:
        return -1.0, {k: 0.0 for k in WEIGHTS}

    details = {
        "clip": clip_similarity(ref_image, gen_image),
        "text": text_similarity(ref_text, gen_text),
        "layout": layout_score(ref_blocks, gen_blocks),
        "color": color_palette_similarity(ref_colors, gen_colors),
        "visual": visual_similarity(ref_image, gen_image),
    }

    raw = sum(WEIGHTS[k] * details[k] for k in WEIGHTS)
    reward = 2.0 * raw - 1.0

    return float(reward), details


# ── Backward-compatible wrapper ──────────────────────────────────────────────

def compute_visual_reward(
    generated_html: str | None,
    reference_image: np.ndarray,
    page: Page,
    size: int = 256,
    reference_html: str = "",
) -> float:
    if generated_html is None:
        return -1.0

    reward, _ = compute_reward(generated_html, reference_html, reference_image, page, size)
    return reward
