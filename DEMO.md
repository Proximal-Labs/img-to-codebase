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
> - Data: Mind2Web

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

### 4B RL (10 batches, single-shot) vs Base — Mind2Web

> **Experiment details:**
> - Model: Qwen 3.5-4B, LoRA rank 32, 10 batches RL (BS=16, GS=2)
> - Reward: Pure SSIM (training), content-gated SSIM (eval)
> - Data: Mind2Web landing pages, 8K token limit

**Resy** — RL adds styled button, Climbing/Top Rated cards
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_00/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_00/turn1.png) | ![rl](eval_output/m2w-simple-batch10/example_00/turn1.png) |

**eBay** — RL gets teal hero section, brand cards layout
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_05/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_05/turn1.png) | ![rl](eval_output/m2w-simple-batch10/example_05/turn1.png) |

**Rentalcars** — RL produces blue gradient + search form (base missed entirely)
| Reference | Base | RL Batch 10 |
|-----------|------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_07/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_07/turn1.png) | ![rl](eval_output/m2w-simple-batch10/example_07/turn1.png) |

| # | Website | Base SSIM | RL SSIM | Delta |
|---|---------|----------|---------|-------|
| 1 | Resy | 0.745 | 0.743 | -0.002 |
| 2 | FoxSports | 0.497 | 0.395 | -0.102 |
| 3 | UnderArmour | 0.533 | 0.429 | -0.104 |
| 4 | IKEA | 0.518 | 0.542 | +0.024 |
| 5 | Yelp | 0.471 | 0.474 | +0.003 |
| 6 | eBay | 0.716 | 0.627 | -0.089 |
| 7 | Carnival | 0.550 | 0.387 | -0.163 |
| 8 | Rentalcars | 0.396 | 0.573 | **+0.177** |
| 9 | Viator | 0.322 | 0.328 | +0.006 |
| 10 | SoundCloud | 0.208 | 0.228 | +0.020 |
| **Avg** | | **0.536** | **0.472** | **-0.064** |

*Mixed results after 10 batches — some sites improve significantly (Rentalcars +0.177) while others regress. Training uses pure SSIM reward but eval uses content-gated SSIM, causing mismatch. Multi-turn agent training (with matching reward) in progress.*

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

Learnings:
**SSIM is an effective reward but is vulnerable to hacking, so custom DOM-based rewards are needed—carefully designed.** While SSIM captures visual similarity, the model can exploit it by generating blank or low-variance pages that artificially inflate scores. DOM-based rewards (like text, layout, color matching) add real signal but must be constructed to never penalize visually correct outputs—custom rewards should complement SSIM, not contradict it. The system should reward correct appearance regardless of HTML structure, while using checks (like pixel variance) to prevent SSIM exploitation.
