# Experiment Log: Screenshot → HTML/CSS RL Training

## Experiment 1: Proof of Concept (Qwen3.5-4B, Pixel Reward)

**Goal:** Get RL training working end-to-end on Tinker.

### Setup
- **Model:** Qwen3.5-4B (smallest available, fast iteration)
- **Dataset:** 175 synthetic HTML snippets (colored divs, buttons, headings, simple layouts) rendered via Playwright
- **Loss:** GRPO + importance sampling
- **Group size:** 4 rollouts per prompt
- **LR:** 4e-5
- **Max tokens:** 512

### Reward Function v1: Pixel-only (SSIM + MSE)
```
reward = 0.6 * pixel_mse_score + 0.4 * ssim_score
# Computed on content-cropped region to avoid white-space domination
# Scaled to [-1, 1]
```

**Problem:** SSIM was dominated by white backgrounds — a red button vs blue button scored ~0.95 because 90% of pixels are identical white.

**Fix:** Crop to content bounding box before comparison. This spread the scores:
- Perfect match: 1.0
- Wrong color: 0.43
- Totally wrong: -0.41

### Results
- **Training:** 21 batches, reward went from -0.32 → +0.45 over ~25 minutes
- **Outcome:** Model clearly learned to produce HTML matching screenshots. Proved the pipeline works.

### Key Decisions
- Disabled thinking mode (`qwen3_5_disable_thinking`) — the model was spending all tokens on chain-of-thought instead of generating HTML
- Started with synthetic data (colored boxes) for simplicity, then switched to WebSight for real websites

---

## Experiment 2: Scaling Up (Qwen3.5-4B, WebSight, PPO)

**Goal:** Train on real website data with better RL algorithms.

### Setup Changes
- **Dataset:** 254 WebSight v0.1 examples (realistic synthetic websites with inline CSS)
- **Loss:** GRPO + PPO (clipped ratio [0.8, 1.2]) — prevents large policy jumps
- **KL penalty:** β=0.05 — prevents mode collapse
- **Group size:** 8 (up from 4) — better advantage estimates
- **Checkpoints:** Every 10 batches via `training_client.save_state()`

### Reward Function v2: Same pixel-only (SSIM + MSE)
```
reward = 0.2 * pixel_mse_score + 0.8 * ssim_score
# Content-cropped, scaled to [-1, 1]
```

### Results
- **Training:** 31 batches, reward from ~0.3 → 0.67
- **Eval (base vs RL):** Base avg 0.252, RL avg 0.332, improvement +0.080, RL wins 6/10
- **KL:** Converged from 0.162 → 0.015 (model became confident without collapsing)
- **No skipped batches** — GROUP_SIZE=8 always produced variance for advantages

### Key Decisions
- PPO clipping stabilized training vs raw importance sampling
- KL penalty prevented reward hacking
- GROUP_SIZE=8 eliminated the "all same reward" skip problem from GROUP_SIZE=4

---

## Experiment 3: Multi-Signal Reward, Larger Model (Qwen3.5-27B)

**Goal:** Better reward function covering text, color, layout, not just pixels. Bigger model.

### Setup Changes
- **Model:** Qwen3.5-27B (7x capacity)
- **Dataset:** 974 WebSight v0.2 + 483 Design2Code (1457 total)
- **Max tokens:** 1024
- **LR:** 4e-5 (same as 4B — too high for 27B)

### Reward Function v3: 8-signal DOM-based
```
reward = weighted sum of:
  0.20 * block_position    (IoU of matched DOM elements via querySelectorAll('*'))
  0.20 * text_content      (per-block fuzzy string match)
  0.10 * bg_color_match    (per-block RGB distance)
  0.05 * text_color_match  (per-block RGB distance)
  0.05 * font_family_match (exact match)
  0.05 * font_size_match   (ratio-based)
  0.20 * clip_similarity   (CLIP ViT-B-32 cosine similarity)
  0.15 * visual_ssim       (pixel SSIM+MSE on content crop)
```

### Problems
1. **Reward was too noisy** — 8 signals pulled in different directions. GRPO couldn't find a clean gradient.
2. **DOM block matching was too strict** — `querySelectorAll('*')` matched every wrapper div. 80 ref blocks vs 40 gen blocks → `match_ratio=0.5` killed all DOM scores by 50%, even when the page looked correct.
3. **Design2Code pages too complex** — full real websites can't be reproduced in 1024 tokens. Added noise to training.
4. **LR too high for 27B** — caused oscillation instead of steady improvement.

### Results
- **Training:** 182 batches (~7 hours), reward oscillated between -0.25 and +0.07
- **20-batch averages** showed the reward actually got *worse* mid-training (batch 81-100: -0.212) then partially recovered
- **Eval:** Base 0.308, RL 0.340, improvement only +0.032. RL wins 6/10.

### Diagnosis
The reward landscape was not smoothly climbable. Small DOM mismatches (extra wrapper divs, different nesting) caused cliff-edge score drops even when the visual output was correct. The model couldn't tell what direction to move.

---

## Experiment 4: Smooth Reward, Tuned Hyperparams (In Progress)

**Goal:** Fix the reward to be smoothly climbable. Each signal should degrade gracefully.

### Setup Changes
- **LR:** 1e-5 (4x lower, appropriate for 27B)
- **Max tokens:** 2048 (let model generate complex pages)
- **Dataset:** WebSight only (removed Design2Code from training, kept for eval)

### Reward Function v4: 5-signal, globally smooth
```
reward = weighted sum of:
  0.25 * clip_similarity     (CLIP ViT-B-32, perceptual anchor)
  0.25 * global_text_match   (ALL visible text compared via SequenceMatcher — not per-block)
  0.20 * layout_score        (meaningful elements only, soft count penalty)
  0.15 * color_palette_match (quantized color histogram overlap — not per-block)
  0.15 * visual_ssim         (pixel SSIM+MSE on content crop)
```

### Key Changes from v3

| Problem in v3 | Fix in v4 |
|---|---|
| `querySelectorAll('*')` matched every div | Filter to meaningful tags only (h1-h6, p, nav, button, etc.) + elements with text/bg color |
| `match_ratio` multiplier killed scores | Soft count penalty: `0.5 + 0.5 * count_ratio` (ranges 0.5-1.0 instead of 0.0-1.0) |
| Per-block text matching inherited block matching noise | Global text comparison — extract ALL visible text, fuzzy match as one string |
| Per-block color matching was noisy | Color palette comparison — quantize colors, compare histograms |
| 8 signals too many, contradictory gradients | 5 signals, each independently smooth |

### Reward Behavior (tested)
```
Perfect match:    +1.000  [all signals 1.0]
Wrong nav color:  +0.839  [text/layout stay 1.0, color/clip drop slightly]
Wrong text:       +0.690  [text drops 0.85, CLIP 0.57, layout stays 1.0]
Missing nav:      +0.385  [layout drops 0.44, everything degrades gracefully]
Extra elements:   +0.761  [small layout penalty, not over-punished]
Slightly off:     +0.802  [smooth gradient toward perfect]
```

No cliff edges. Each error type degrades the right signals without killing the whole score.

### Results (4-batch smoke test)
- **Training:** 4 batches, reward 0.058 → 0.190
- **Eval:** Base 0.231, RL 0.200, improvement -0.031, RL wins 5/10
- **Assessment:** Essentially tied after 4 batches. Reward is smooth and climbable (all positive, no -1.0 catastrophes) but model needs more training time to separate from base.

---

## Experiment 5: Tall Screenshots (512x1536 Viewport)

**Goal:** Show the model full page content instead of cutting off at 512px.

### Setup Changes
- **Viewport:** 512x1536 (3x taller) — shows nav, hero, body, footer
- **Everything else same as Experiment 4**

### Key Insight
The model was being asked to reproduce content it couldn't see. With 512x512 viewport, most pages were cut off — the reference HTML had sections below the fold that never appeared in the screenshot. The model was penalized for content it literally had no visual signal for.

### Results (8-batch run)
- **Training:** 8 batches, reward 0.116 → 0.145 (consistently positive)
- **Eval:** Base 0.586, RL 0.585, improvement -0.000, RL wins 6/10
- **Absolute scores jumped massively** — base went from 0.231 → 0.586 just from showing more content. The model was already decent; it just couldn't see the full page before.

### Assessment
The tall viewport was the single biggest improvement to absolute quality. RL essentially tied with base after 8 batches but won 6/10 matchups — needs more training to compound.

---

## Experiment 6: Desktop Viewport + Curriculum (In Progress)

**Goal:** Natural desktop layout (1024x768), curriculum learning (easy→hard), include Design2Code pages.

### Setup Changes
- **Viewport:** 1024x768 (standard desktop, proper horizontal layout)
- **Dataset:** 974 WebSight + 37 Design2Code (<8K chars) = 1011 examples
- **Curriculum:** Dataset sorted by HTML length — shortest pages first, longest last. Model learns simple layouts (nav + heading) before complex ones (multi-section pages with forms, cards, etc.)
- **MAX_TOKENS:** 4096 (up from 2048, supports longer D2C pages)
- **LR:** 1e-5
- **Design2Code** filtered to pages <8K chars (37 of 483 qualify) — the rest are 60K+ chars and can't be generated in 4096 tokens

### Why Curriculum
Without curriculum, the model sees a random mix of easy and hard pages. It wastes gradient signal on hard pages it can't solve yet, while also not spending enough time on easy pages to nail the fundamentals. With curriculum:
- Batches 0-28: WebSight pages (447-2000 chars) — learn basic HTML structure, colors, fonts
- Batches 29-30: Design2Code pages (4000-8000 chars) — apply skills to real websites

### Why 1024x768
The 512px width was squishing layouts into a single column. Real websites are designed for 1024-1440px width — nav bars should be horizontal, content should use the full width, sidebars should be visible. At 1024px the model sees the intended layout, not a mobile-like rendering.

### Status
Running 30-batch curriculum training. 1011 examples, checkpoint at batch 15, auto-eval at the end.

### Reward Function (unchanged from v4)
```
0.25 * CLIP perceptual similarity
0.25 * global text match (SequenceMatcher on all visible text)
0.20 * layout score (meaningful DOM elements, soft count penalty)
0.15 * color palette similarity (quantized histogram overlap)
0.15 * visual SSIM+MSE (content-cropped)
```
