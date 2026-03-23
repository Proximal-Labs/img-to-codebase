# Screenshot → HTML: Research Demo

## 1. Datasets Explored

### WebSight v2 — Synthetic Websites
Synthetic but realistic pages with Tailwind CSS.

![websight1](data/screenshots_1024/0001.png)
![websight2](data/screenshots_1024/0050.png)
![websight3](data/screenshots_1024/0100.png)

### Design2Code — Real Websites
484 real webpages from C4 corpus.

![d2c1](data/design2code/screenshots/0010.png)
![d2c2](data/design2code/screenshots/0050.png)
![d2c3](data/design2code/screenshots/0100.png)

### Mind2Web — Actual Live Websites
Real screenshots from Resy, eBay, ESPN, IKEA, United Airlines, etc.

![m2w-resy](eval_output/single_image_rl/4b-base-simple/example_00/ref.png)
![m2w-foxsports](eval_output/single_image_rl/4b-base-simple/example_01/ref.png)
![m2w-ikea](eval_output/single_image_rl/4b-base-simple/example_03/ref.png)
![m2w-ebay](eval_output/single_image_rl/4b-base-simple/example_05/ref.png)
![m2w-soundcloud](eval_output/single_image_rl/4b-base-simple/example_09/ref.png)

---

## 2. Task 1: One-Shot Screenshot → HTML

### 27B on WebSight + Design2Code (90 batches, best early result)

> **Experiment details:**
> - Model: Qwen 3.5-27B (LoRA rank 32)
> - Reward: DOM-based (0.30 text + 0.30 layout + 0.20 color + 0.20 visual SSIM)
> - Context: 1024 tokens
> - Data: 1011 examples (974 WebSight v2 + 37 Design2Code), curriculum ordered
> - Viewport: 1024x768
> - Result: **+0.231 improvement, 9/10 wins** at batch 75 checkpoint

| Reference | Base (27B untrained) | RL (batch 75) | Delta |
|-----------|---------------------|---------------|-------|
| ![ref](eval_output/single_image_rl/exp9-batch75/example_02/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_02/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_02/rl.png) | +0.199 |
| ![ref](eval_output/single_image_rl/exp9-batch75/example_06/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_06/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_06/rl.png) | +0.749 |
| ![ref](eval_output/single_image_rl/exp9-batch75/example_09/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_09/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_09/rl.png) | +0.469 |
| ![ref](eval_output/single_image_rl/exp9-batch75/example_00/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_00/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_00/rl.png) | +0.079 |
| ![ref](eval_output/single_image_rl/exp9-batch75/example_03/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_03/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_03/rl.png) | +0.035 |

### 4B After 10 Batches RL Training

> **Experiment details:**
> - Model: Qwen 3.5-4B (LoRA rank 32)
> - Reward: 0.60 SSIM (content-cropped 256x256) + 0.25 text match + 0.15 color match
> - Context: 2048 tokens max
> - Data: WebSight v2 (not Mind2Web — this was before we switched datasets)
> - Note: 2K token limit caused HTML cutoff on some examples. Reward used content-cropped SSIM which diverged from full-viewport SSIM.

**SoundCloud**
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_09/ref.png) | ![base](eval_output/single_image_rl/4b-base-simple/example_09/gen.png) | ![rl](eval_output/single_image_rl/4b-simple-batch10/example_09/gen.png) |

**Resy**
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_00/ref.png) | ![base](eval_output/single_image_rl/4b-base-simple/example_00/gen.png) | ![rl](eval_output/single_image_rl/4b-simple-batch10/example_00/gen.png) |

**Under Armour**
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_02/ref.png) | ![base](eval_output/single_image_rl/4b-base-simple/example_02/gen.png) | ![rl](eval_output/single_image_rl/4b-simple-batch10/example_02/gen.png) |

**eBay**
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_05/ref.png) | ![base](eval_output/single_image_rl/4b-base-simple/example_05/gen.png) | ![rl](eval_output/single_image_rl/4b-simple-batch10/example_05/gen.png) |

**Carnival**
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_06/ref.png) | ![base](eval_output/single_image_rl/4b-base-simple/example_06/gen.png) | ![rl](eval_output/single_image_rl/4b-simple-batch10/example_06/gen.png) |

| Site | Base SSIM | RL SSIM | SSIM Delta | Base Reward | RL Reward | Reward Delta |
|------|-----------|---------|------------|-------------|-----------|--------------|
| Resy | 0.816 | **0.849** | +0.033 | 0.632 | **0.699** | +0.067 |
| FoxSports | 0.473 | 0.381 | -0.092 | -0.055 | -0.237 | -0.182 |
| UnderArmour | 0.520 | **0.527** | +0.007 | 0.039 | -0.040 | -0.079 |
| IKEA | 0.580 | 0.480 | -0.100 | 0.160 | -0.040 | -0.200 |
| Yelp | 0.468 | 0.429 | -0.039 | -0.063 | -0.142 | -0.079 |
| eBay | 0.727 | 0.603 | -0.124 | 0.453 | 0.207 | -0.246 |
| Carnival | 0.588 | 0.463 | -0.125 | 0.176 | -0.074 | -0.250 |
| Rentalcars | 0.564 | 0.478 | -0.086 | 0.128 | -0.044 | -0.172 |
| Viator | 0.334 | **0.376** | +0.042 | -0.333 | -0.248 | +0.085 |
| SoundCloud | 0.206 | **0.545** | +0.339 | -0.588 | **0.090** | +0.678 |
| **Average** | **0.528** | **0.513** | **-0.015** | **0.055** | **0.017** | **-0.038** |

*Note: batch 10 model was trained with 2K token limit (HTML cutoff issues) and without blank page penalty. Later runs fix both.*

---

## 3. Task 2: Multi-Turn Analyze-Fix Agent

Model generates HTML → we render → model sees target vs output side-by-side → analyzes differences → fixes.

### 4B Base (2 turns, analyze-fix) — Mind2Web Real Websites

> **Experiment details:**
> - Model: Qwen 3.5-4B base (no RL training)
> - Eval: 10 Mind2Web real websites, 2 turns with analyze-fix
> - Reference: Original website screenshots (not DOM re-renders)
> - Result: **Avg SSIM 0.468, Avg reward -0.658**

**Resy** (SSIM 0.753)
| Reference | Base 4B Output |
|-----------|---------------|
| ![ref](eval_output/m2w-4b-base-2turns-v2/example_00/ref-render.png) | ![gen](eval_output/m2w-4b-base-2turns-v2/example_00/turn1.png) |

**eBay** (SSIM 0.630)
| Reference | Base 4B Output |
|-----------|---------------|
| ![ref](eval_output/m2w-4b-base-2turns-v2/example_05/ref-render.png) | ![gen](eval_output/m2w-4b-base-2turns-v2/example_05/turn1.png) |

| Example | Turn 1 SSIM | Turn 2 SSIM | Turn 1 Reward | Turn 2 Reward |
|---------|------------|------------|---------------|---------------|
| 1 (Resy) | 0.753 | 0.753 | -0.768 | -0.769 |
| 2 | 0.409 | 0.356 | -0.710 | -0.744 |
| 3 | 0.470 | 0.468 | -0.658 | -0.688 |
| 4 | 0.496 | 0.496 | -0.472 | -0.472 |
| 5 | 0.324 | 0.324 | -0.679 | -0.679 |
| 6 (eBay) | 0.630 | 0.628 | -0.858 | -0.830 |
| 7 | 0.551 | 0.552 | -0.879 | -0.878 |
| 8 | 0.528 | 0.545 | -0.347 | -0.334 |
| 9 | 0.352 | 0.352 | -0.422 | -0.422 |
| 10 | 0.205 | 0.206 | -0.815 | -0.763 |
| **Avg** | **0.472** | **0.468** | **-0.661** | **-0.658** |

*Note: Analyze-fix barely helps on base model (+0.003 reward). The model isn't trained to use visual feedback. RL training in progress to improve this.*

### GPT-5.4: Analyze-Fix vs Naive (10 turns)
```
Naive:        0.444 → peaked 0.490 → REGRESSED to 0.430 by turn 10
Analyze-fix:  0.442 → 0.520 → held at 0.509 (no regression)
```

The model's self-analysis was specific and accurate:
> "heading font size is too large, paragraph line-height is too tall,
> font weight too heavy, paragraph width too narrow"

---

## 4. Task 3: Interactive Flow (Action Sequences)

Model sees multiple screenshots showing a user flow, generates interactive HTML with JavaScript.

### Example: Budget Car Rental User Flow (6 steps)

The reference flow shows a user booking a luxury car rental — the page changes completely at each step:

| Step 1 — Landing page | Step 3 — Location search | Step 5 — Car selection |
|----------------------|--------------------------|----------------------|
| ![step0](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_00/step_0_ref.png) | ![step2](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_00/step_2_ref.png) | ![step4](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_00/step_4_ref.png) |

### GPT-5.4 Generated Landing Page (SSIM 0.782)

The model saw all flow screenshots and generated HTML that matches the initial state:

| Reference | GPT-5.4 Generated |
|-----------|-------------------|
| ![ref](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_00/step_0_ref.png) | ![gen](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_00/step_0_gen.png) |

### GPT-5.4 on Rotten Tomatoes (SSIM 0.470)

| Reference | GPT-5.4 Generated |
|-----------|-------------------|
| ![ref](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_02/step_0_ref.png) | ![gen](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_02/step_0_gen.png) |

### GPT-5.4 on Resy (SSIM 0.817)

| Reference | GPT-5.4 Generated |
|-----------|-------------------|
| ![ref](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_04/step_0_ref.png) | ![gen](eval_output/flow_interactive/mind2web-gpt54-flow-16k/task_04/step_0_gen.png) |

---

---

## 5. Frontier Model Comparisons
| ![ref](eval_output/frontier_baselines/gpt54-hard-d2c/task_00/reference.png) | ![gen](eval_output/frontier_baselines/gpt54-hard-d2c/task_00/turn1.png) | 0.930 |
| ![ref](eval_output/frontier_baselines/gpt54-hard-d2c/task_01/reference.png) | ![gen](eval_output/frontier_baselines/gpt54-hard-d2c/task_01/turn1.png) | 0.776 |
| ![ref](eval_output/frontier_baselines/gpt54-hard-d2c/task_05/reference.png) | ![gen](eval_output/frontier_baselines/gpt54-hard-d2c/task_05/turn1.png) | 0.941 |

### The SSIM-Perfect-But-Reward-Broken Discovery
GPT-5.4 produced SSIM 0.991 (pixel-perfect) but our DOM reward scored -0.5. This led us to switch to pure SSIM reward.

---

## 7. Key Findings

1. **SSIM is the right reward** — DOM comparison penalized pixel-perfect outputs
2. **Analyze-then-fix prevents regression** — splitting "what's wrong" from "fix it" works
3. **Output is always short** — 100K char pages reproduced in 1-4K chars
4. **Frontier models generate interactive JS** — GPT-5.4: 18-22 handlers, Opus: full multi-page apps
5. **Test your harness** — our Playwright wasn't clicking buttons (selector bug)

## 8. Challenges

| Challenge | Solution |
|-----------|----------|
| DOM reward broke on pixel-perfect outputs | Pure SSIM reward |
| Pure SSIM reward hacking (blank pages score high) | Penalize low pixel variance (std < 10) |
| Playwright selectors=None (74% of clicks failed) | Text-based matching |
| Mind2Web raw_html has no CSS (DOM dump) | Compare against original screenshots |
| 27B multi-image: 4.5 hrs/batch | One-shot single image, parallel sampling |
| HTML cutoff (white screens) | 16K token limit |
| Viewport sizes causing bad comparisons | 1024x768 or 1280x720 standard |

## 9. Next Steps

1. **SSIM is a great reward, but you need to prevent hacking.** The model learned to generate blank pages that score high SSIM against light backgrounds. Simple fix (pixel variance check) works but more sophisticated checks may be needed at scale.

2. **Custom rewards are valuable but must be carefully constructed.** DOM-based rewards captured real signal (text match, layout, colors) but penalized visually identical outputs when the internals differed. The lesson: custom rewards should never contradict SSIM — if it looks right, the reward should be high regardless of how the HTML is structured internally.

3. **Scale training on harder datasets.** Mind2Web real websites are the right difficulty level. WebSight/D2C are too easy — models already score well without RL.

4. **Agentic multi-turn with analyze-fix at inference time.** Train one-shot for speed, add the analyze-fix loop at inference time for quality. The analyze step costs one extra API call but prevents regression.

5. **Interactive flow training needs faster infrastructure.** 27B with multi-image prompts is 4.5 hrs/batch. Either use smaller models for flow training or wait for faster inference.
