"""
Multi-signal reward function for screenshot-to-HTML RL.

Designed to be smoothly climbable — each signal degrades gracefully.
Pure DOM comparison — no model inference (CLIP removed).

Reward signals:
  0.30 - global text content match (all visible text compared)
  0.30 - layout / block position matching (meaningful elements only)
  0.20 - color palette similarity (quantized histogram overlap)
  0.20 - visual SSIM+MSE (pixel-level)
"""

import io
import re
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
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
    try:
        page.wait_for_load_state("networkidle", timeout=3000)
    except Exception:
        page.wait_for_timeout(200)


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


# ── Combined DOM extraction (single JS call) ─────────────────────────────────

MEANINGFUL_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "a", "button", "input",
    "img", "nav", "header", "footer", "main", "section", "article",
    "li", "td", "th", "label", "span", "textarea", "select",
}


def extract_dom_info(page: Page) -> dict:
    """
    Extract all DOM info in a single JS evaluate call.
    Returns {text, blocks, colors} — no redundant round-trips.
    """
    tags_list = list(MEANINGFUL_TAGS)
    return page.evaluate("""(tagsList) => {
        const meaningfulTags = new Set(tagsList);
        const blocks = [];
        const colors = [];
        const els = document.querySelectorAll('*');

        for (const el of els) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 5 || rect.height < 5) continue;

            const tag = el.tagName.toLowerCase();
            const style = getComputedStyle(el);

            // Skip invisible elements
            if (style.display === 'none' || style.visibility === 'hidden' ||
                parseFloat(style.opacity) === 0) continue;

            // Colors (from all visible elements)
            if (rect.width >= 10 && rect.height >= 10) {
                const bg = style.backgroundColor;
                if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
                    const m = bg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    if (m) colors.push([parseInt(m[1]), parseInt(m[2]), parseInt(m[3])]);
                }
                const fg = style.color;
                if (fg) {
                    const m = fg.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/);
                    if (m) colors.push([parseInt(m[1]), parseInt(m[2]), parseInt(m[3])]);
                }
            }

            // Blocks (meaningful elements only)
            if (rect.width >= window.innerWidth * 0.99 && rect.height >= window.innerHeight * 0.99) continue;

            const hasBg = style.backgroundColor !== 'rgba(0, 0, 0, 0)' && style.backgroundColor !== 'transparent';
            const hasBorder = style.borderWidth && style.borderWidth !== '0px';

            let directText = '';
            for (const node of el.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    directText += node.textContent;
                }
            }
            directText = directText.trim();

            if (meaningfulTags.has(tag) || directText.length > 0 || hasBg || hasBorder) {
                blocks.push({
                    tag: tag,
                    x: rect.x,
                    y: rect.y,
                    w: rect.width,
                    h: rect.height,
                    text: directText.substring(0, 200),
                    fontSize: parseFloat(style.fontSize) || 0,
                    fontWeight: style.fontWeight || '400',
                    color: style.color || '',
                });
            }
        }

        return {
            text: document.body ? document.body.innerText.trim() : '',
            blocks: blocks,
            colors: colors,
        };
    }""", tags_list)


def extract_ref_info(page: Page, reference_html: str, size: int = 256) -> dict:
    """
    Extract all reference info in one shot. Call once per prompt, reuse
    across all rollouts in the group.

    Returns {text, blocks, colors, image}.
    """
    render_html(page, reference_html)
    dom = extract_dom_info(page)
    screenshot_bytes = page.screenshot()
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB").resize((size, size))
    dom["image"] = np.array(img)
    return dom


def extract_gen_info(page: Page, generated_html: str, size: int = 256) -> dict:
    """
    Render generated HTML once, extract DOM + screenshot in same page load.
    """
    render_html(page, generated_html)
    dom = extract_dom_info(page)
    screenshot_bytes = page.screenshot()
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB").resize((size, size))
    dom["image"] = np.array(img)
    return dom


# ── Comparison functions ─────────────────────────────────────────────────────

def text_similarity(ref_text: str, gen_text: str) -> float:
    """Global text content similarity."""
    if not ref_text and not gen_text:
        return 1.0
    if not ref_text or not gen_text:
        return 0.0
    return SequenceMatcher(None, ref_text.lower(), gen_text.lower()).ratio()


def _parse_css_color(color_str: str) -> tuple[int, int, int] | None:
    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color_str)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def _color_dist(c1, c2) -> float:
    if c1 is None or c2 is None:
        return 1.0
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5 / (255 ** 2 * 3) ** 0.5


def styled_text_score(ref_blocks: list[dict], gen_blocks: list[dict]) -> float:
    """
    Compare text content paired with styling (font size, weight, color).
    Each text-bearing block is matched by content similarity, then
    styling similarity adds a bonus.
    """
    ref_texts = [b for b in ref_blocks if b.get("text")]
    gen_texts = [b for b in gen_blocks if b.get("text")]

    if not ref_texts and not gen_texts:
        return 1.0
    if not ref_texts or not gen_texts:
        return 0.0

    used_gen = set()
    scores = []

    for ref in ref_texts:
        best_score = 0.0
        best_idx = -1

        for j, gen in enumerate(gen_texts):
            if j in used_gen:
                continue
            # Text content match
            txt_sim = SequenceMatcher(None, ref["text"].lower(), gen["text"].lower()).ratio()
            if txt_sim > best_score:
                best_score = txt_sim
                best_idx = j

        if best_idx >= 0 and best_score > 0.3:
            gen = gen_texts[best_idx]
            used_gen.add(best_idx)

            # Style bonus: font size, weight, color
            style_scores = []

            # Font size ratio
            ref_fs = ref.get("fontSize", 0)
            gen_fs = gen.get("fontSize", 0)
            if ref_fs > 0 and gen_fs > 0:
                style_scores.append(min(ref_fs, gen_fs) / max(ref_fs, gen_fs))

            # Font weight match
            style_scores.append(1.0 if ref.get("fontWeight") == gen.get("fontWeight") else 0.5)

            # Text color
            ref_c = _parse_css_color(ref.get("color", ""))
            gen_c = _parse_css_color(gen.get("color", ""))
            style_scores.append(1.0 - _color_dist(ref_c, gen_c))

            style_bonus = sum(style_scores) / len(style_scores) if style_scores else 1.0

            # Blend: 70% text content, 30% style match
            scores.append(0.7 * best_score + 0.3 * style_bonus)
        else:
            scores.append(0.0)

    # Soft penalty for unmatched
    count_ratio = min(len(ref_texts), len(gen_texts)) / max(len(ref_texts), len(gen_texts))
    count_penalty = 0.5 + 0.5 * count_ratio

    return float((sum(scores) / len(scores)) * count_penalty)


def _size_similarity(a: dict, b: dict) -> float:
    """Compare width and height similarity (ratio-based)."""
    w_sim = min(a["w"], b["w"]) / max(a["w"], b["w"]) if max(a["w"], b["w"]) > 0 else 1.0
    h_sim = min(a["h"], b["h"]) / max(a["h"], b["h"]) if max(a["h"], b["h"]) > 0 else 1.0
    return (w_sim + h_sim) / 2


TAG_WEIGHTS = {
    "h1": 2.0, "h2": 1.8, "h3": 1.6,
    "button": 1.8, "input": 1.8, "textarea": 1.6, "select": 1.6,
    "a": 1.5, "img": 1.5,
    "nav": 1.3, "header": 1.3, "footer": 1.3,
}


def layout_score(ref_blocks: list[dict], gen_blocks: list[dict]) -> float:
    """
    Layout comparison using size similarity + relative vertical ordering.
    Robust to cascading offset errors from earlier elements.
    Important elements (headings, buttons, inputs) are weighted higher.
    """
    if not ref_blocks and not gen_blocks:
        return 1.0
    if not ref_blocks or not gen_blocks:
        return 0.0

    # Sort both by vertical position
    ref_sorted = sorted(ref_blocks, key=lambda b: (b["y"], b["x"]))
    gen_sorted = sorted(gen_blocks, key=lambda b: (b["y"], b["x"]))

    # Match by best size + text overlap, weighted by tag importance
    used_gen = set()
    weighted_scores = []
    weights = []

    for ref in ref_sorted:
        best_score = 0.0
        best_idx = -1

        for j, gen in enumerate(gen_sorted):
            if j in used_gen:
                continue

            # Size similarity
            size_sim = _size_similarity(ref, gen)

            # Text overlap bonus (if both have text)
            text_sim = 0.0
            if ref.get("text") and gen.get("text"):
                text_sim = SequenceMatcher(None, ref["text"].lower(), gen["text"].lower()).ratio()

            score = 0.5 * size_sim + 0.5 * text_sim if (ref.get("text") or gen.get("text")) else size_sim

            if score > best_score:
                best_score = score
                best_idx = j

        tag_weight = TAG_WEIGHTS.get(ref.get("tag", ""), 1.0)
        weighted_scores.append(best_score * tag_weight)
        weights.append(tag_weight)
        if best_idx >= 0 and best_score > 0.2:
            used_gen.add(best_idx)

    avg_match = sum(weighted_scores) / sum(weights)

    # Check vertical ordering consistency
    # For matched pairs, verify they appear in the same relative order
    ordering_score = 1.0
    if len(used_gen) >= 2:
        matched_gen_indices = sorted(used_gen)
        # If gen indices are monotonically increasing, ordering is preserved
        inversions = sum(1 for i in range(len(matched_gen_indices) - 1)
                        if matched_gen_indices[i] > matched_gen_indices[i + 1])
        ordering_score = 1.0 - (inversions / max(len(matched_gen_indices) - 1, 1))

    count_ratio = min(len(ref_blocks), len(gen_blocks)) / max(len(ref_blocks), len(gen_blocks))
    count_penalty = 0.5 + 0.5 * count_ratio

    return float(avg_match * ordering_score * count_penalty)


def _quantize_color(c, step: int = 32):
    return (c[0] // step * step, c[1] // step * step, c[2] // step * step)


def color_palette_similarity(ref_colors: list, gen_colors: list) -> float:
    if not ref_colors and not gen_colors:
        return 1.0
    if not ref_colors or not gen_colors:
        return 0.0

    ref_hist = Counter(_quantize_color(tuple(c)) for c in ref_colors)
    gen_hist = Counter(_quantize_color(tuple(c)) for c in gen_colors)

    all_colors = set(ref_hist.keys()) | set(gen_hist.keys())
    intersection = sum(min(ref_hist.get(c, 0), gen_hist.get(c, 0)) for c in all_colors)
    total = max(sum(ref_hist.values()), sum(gen_hist.values()))

    return float(intersection / total) if total > 0 else 0.0


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
    "text": 0.20,          # global text content match
    "styled_text": 0.15,   # text + font size/weight/color match
    "layout": 0.25,        # element sizes + relative ordering
    "color": 0.20,         # color palette
    "visual": 0.20,        # pixel SSIM
}


def compute_reward_from_info(ref_info: dict, gen_info: dict) -> tuple[float, dict]:
    """
    Compute reward from pre-extracted DOM info.
    Both ref_info and gen_info come from extract_dom_info / extract_ref_info.
    """
    details = {
        "text": text_similarity(ref_info["text"], gen_info["text"]),
        "styled_text": styled_text_score(ref_info["blocks"], gen_info["blocks"]),
        "layout": layout_score(ref_info["blocks"], gen_info["blocks"]),
        "color": color_palette_similarity(ref_info["colors"], gen_info["colors"]),
        "visual": visual_similarity(ref_info["image"], gen_info["image"]),
    }

    raw = sum(WEIGHTS[k] * details[k] for k in WEIGHTS)
    reward = 2.0 * raw - 1.0

    return float(reward), details


def compute_reward(
    generated_html: str | None,
    reference_html: str,
    ref_image: np.ndarray,
    page: Page,
    size: int = 256,
) -> tuple[float, dict]:
    """
    Compute reward (non-cached path, used by eval).
    For training, prefer extract_ref_info + compute_reward_from_info.
    """
    if generated_html is None:
        return -1.0, {k: 0.0 for k in WEIGHTS}

    try:
        ref_info = extract_ref_info(page, reference_html, size=size)
        gen_info = extract_gen_info(page, generated_html, size=size)
    except Exception:
        return -1.0, {k: 0.0 for k in WEIGHTS}

    return compute_reward_from_info(ref_info, gen_info)


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
