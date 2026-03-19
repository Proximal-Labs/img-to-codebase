"""
Multi-signal reward function for screenshot-to-HTML RL.

Combines DOM-level signals (block matching, text, color, font) with
image-level signals (CLIP perceptual similarity, SSIM pixel similarity).

Reward weights:
  0.20 - block position (IoU of matched DOM elements)
  0.20 - text content (fuzzy string match)
  0.10 - background color match
  0.05 - text color match
  0.05 - font family match
  0.05 - font size match
  0.20 - CLIP similarity (perceptual)
  0.15 - visual SSIM+MSE (pixel-level)
"""

import io
import re
from difflib import SequenceMatcher

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import Page

from config import VIEWPORT_W, VIEWPORT_H


# ── HTML extraction ───────────────────────────────────────────────────────────

def extract_html_from_response(text: str) -> str | None:
    """Extract HTML from a model response, handling thinking blocks and various formats."""
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


def render_html_to_file(page: Page, html_snippet: str | None, save_path: str) -> bool:
    if html_snippet is None:
        Image.new("RGB", (VIEWPORT_W, VIEWPORT_H), (240, 240, 240)).save(save_path)
        return False
    try:
        render_html(page, html_snippet)
        page.wait_for_timeout(100)
        page.screenshot(path=save_path)
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
    """Lazy-load CLIP model (cached)."""
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
    """Compute CLIP cosine similarity between two images (numpy arrays)."""
    model, preprocess = _get_clip()
    pil_a = Image.fromarray(img_a)
    pil_b = Image.fromarray(img_b)

    with torch.no_grad():
        tensor_a = preprocess(pil_a).unsqueeze(0)
        tensor_b = preprocess(pil_b).unsqueeze(0)
        feat_a = model.encode_image(tensor_a)
        feat_b = model.encode_image(tensor_b)
        feat_a = feat_a / feat_a.norm(dim=-1, keepdim=True)
        feat_b = feat_b / feat_b.norm(dim=-1, keepdim=True)
        sim = (feat_a @ feat_b.T).item()

    return float(sim)


# ── DOM block extraction ─────────────────────────────────────────────────────

def extract_dom_blocks(page: Page) -> list[dict]:
    """Extract visible DOM elements with bounding boxes and computed styles."""
    blocks = page.evaluate("""() => {
        const blocks = [];
        const els = document.querySelectorAll('*');
        for (const el of els) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 5 || rect.height < 5) continue;
            if (rect.width >= window.innerWidth && rect.height >= window.innerHeight) continue;

            const style = getComputedStyle(el);
            const text = el.innerText || '';
            // Only take direct text, not from children
            let directText = '';
            for (const node of el.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    directText += node.textContent;
                }
            }

            blocks.push({
                tag: el.tagName.toLowerCase(),
                x: rect.x,
                y: rect.y,
                w: rect.width,
                h: rect.height,
                bgColor: style.backgroundColor,
                color: style.color,
                fontFamily: style.fontFamily,
                fontSize: parseFloat(style.fontSize) || 0,
                text: directText.trim().substring(0, 200),
                fullText: text.trim().substring(0, 500),
            });
        }
        return blocks;
    }""")
    return blocks


# ── Block matching ────────────────────────────────────────────────────────────

def _iou(a: dict, b: dict) -> float:
    """Intersection over Union of two bounding boxes."""
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _parse_color(color_str: str) -> tuple[int, int, int] | None:
    """Parse CSS color string like 'rgb(255, 0, 0)' or 'rgba(...)' to RGB tuple."""
    match = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+)", color_str)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _color_distance(c1: tuple[int, int, int] | None, c2: tuple[int, int, int] | None) -> float:
    """Normalized color distance (0 = identical, 1 = maximally different)."""
    if c1 is None or c2 is None:
        return 1.0
    dist = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
    max_dist = (255**2 * 3) ** 0.5  # ~441
    return dist / max_dist


def _text_similarity(a: str, b: str) -> float:
    """Fuzzy text similarity (0-1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def match_blocks(ref_blocks: list[dict], gen_blocks: list[dict]) -> dict:
    """
    Match DOM blocks and compute per-signal scores.

    Returns dict with scores for: position, text, bg_color, text_color, font_family, font_size.
    All scores are in [0, 1].
    """
    if not ref_blocks:
        return {
            "position": 1.0 if not gen_blocks else 0.5,
            "text": 1.0 if not gen_blocks else 0.5,
            "bg_color": 1.0 if not gen_blocks else 0.5,
            "text_color": 1.0 if not gen_blocks else 0.5,
            "font_family": 1.0 if not gen_blocks else 0.5,
            "font_size": 1.0 if not gen_blocks else 0.5,
        }

    # Greedy matching: for each ref block, find best gen block by IoU
    used_gen = set()
    matches = []

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

        if best_idx >= 0 and best_iou > 0.05:  # minimum IoU threshold
            matches.append((ref, gen_blocks[best_idx], best_iou))
            used_gen.add(best_idx)

    n_ref = len(ref_blocks)
    match_ratio = len(matches) / n_ref if n_ref > 0 else 0.0

    if not matches:
        return {
            "position": 0.0,
            "text": 0.0,
            "bg_color": 0.0,
            "text_color": 0.0,
            "font_family": 0.0,
            "font_size": 0.0,
        }

    # Score each signal across matched pairs
    position_scores = []
    text_scores = []
    bg_color_scores = []
    text_color_scores = []
    font_family_scores = []
    font_size_scores = []

    for ref, gen, iou in matches:
        position_scores.append(iou)

        text_scores.append(_text_similarity(ref.get("text", ""), gen.get("text", "")))

        ref_bg = _parse_color(ref.get("bgColor", ""))
        gen_bg = _parse_color(gen.get("bgColor", ""))
        bg_color_scores.append(1.0 - _color_distance(ref_bg, gen_bg))

        ref_tc = _parse_color(ref.get("color", ""))
        gen_tc = _parse_color(gen.get("color", ""))
        text_color_scores.append(1.0 - _color_distance(ref_tc, gen_tc))

        # Font family: check if any family matches
        ref_fonts = set(f.strip().strip("'\"").lower() for f in ref.get("fontFamily", "").split(","))
        gen_fonts = set(f.strip().strip("'\"").lower() for f in gen.get("fontFamily", "").split(","))
        font_family_scores.append(1.0 if ref_fonts & gen_fonts else 0.0)

        # Font size: ratio-based
        ref_fs = ref.get("fontSize", 0)
        gen_fs = gen.get("fontSize", 0)
        if ref_fs > 0 and gen_fs > 0:
            ratio = min(ref_fs, gen_fs) / max(ref_fs, gen_fs)
            font_size_scores.append(ratio)
        elif ref_fs == 0 and gen_fs == 0:
            font_size_scores.append(1.0)
        else:
            font_size_scores.append(0.0)

    # Weight by match ratio (penalize missing blocks)
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "position": avg(position_scores) * match_ratio,
        "text": avg(text_scores) * match_ratio,
        "bg_color": avg(bg_color_scores) * match_ratio,
        "text_color": avg(text_color_scores) * match_ratio,
        "font_family": avg(font_family_scores) * match_ratio,
        "font_size": avg(font_size_scores) * match_ratio,
    }


# ── Visual similarity (SSIM + MSE) ───────────────────────────────────────────

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
    """SSIM + MSE on content-cropped region, returns score in [0, 1]."""
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

# Reward weights
WEIGHTS = {
    "position": 0.20,
    "text": 0.20,
    "bg_color": 0.10,
    "text_color": 0.05,
    "font_family": 0.05,
    "font_size": 0.05,
    "clip": 0.20,
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
    Compute multi-signal reward comparing generated HTML to reference.

    Args:
        generated_html: model output (or None if extraction failed)
        reference_html: ground-truth HTML snippet
        ref_image: reference screenshot as numpy array (for visual/CLIP)
        page: Playwright page for rendering
        size: image size for visual comparison

    Returns:
        (reward, details) where reward is in [-1, 1] and details is a dict of per-signal scores.
    """
    if generated_html is None:
        details = {k: 0.0 for k in WEIGHTS}
        return -1.0, details

    try:
        # Render generated HTML
        render_html(page, generated_html)
        gen_image = render_html_to_image(page, generated_html, size=size)

        # Extract DOM blocks from both
        render_html(page, reference_html)
        ref_blocks = extract_dom_blocks(page)

        render_html(page, generated_html)
        gen_blocks = extract_dom_blocks(page)

    except Exception:
        details = {k: 0.0 for k in WEIGHTS}
        return -1.0, details

    # DOM-level signals
    block_scores = match_blocks(ref_blocks, gen_blocks)

    # Image-level signals
    clip_score = clip_similarity(ref_image, gen_image)
    vis_score = visual_similarity(ref_image, gen_image)

    details = {
        "position": block_scores["position"],
        "text": block_scores["text"],
        "bg_color": block_scores["bg_color"],
        "text_color": block_scores["text_color"],
        "font_family": block_scores["font_family"],
        "font_size": block_scores["font_size"],
        "clip": clip_score,
        "visual": vis_score,
    }

    # Weighted sum
    raw = sum(WEIGHTS[k] * details[k] for k in WEIGHTS)

    # Scale to [-1, 1]
    reward = 2.0 * raw - 1.0

    return float(reward), details


# ── Backward-compatible wrapper ──────────────────────────────────────────────

def compute_visual_reward(
    generated_html: str | None,
    reference_image: np.ndarray,
    page: Page,
    size: int = 256,
    reference_html: str | None = None,
) -> float:
    """
    Compute reward. Uses full multi-signal if reference_html is provided,
    otherwise falls back to visual-only (SSIM+MSE).
    """
    if generated_html is None:
        return -1.0

    if reference_html is not None:
        reward, _ = compute_reward(generated_html, reference_html, reference_image, page, size)
        return reward

    # Fallback: visual only
    try:
        gen_image = render_html_to_image(page, generated_html, size=size)
    except Exception:
        return -1.0

    score = visual_similarity(reference_image, gen_image)
    return float(2.0 * score - 1.0)
