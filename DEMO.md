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

---

## 3. Task 2: Single-Shot Screenshot → HTML (4B on Mind2Web)

Base 4B vs RL-trained 4B (10 batches). Both single-shot (1 turn, no analyze-fix). Same 10 Mind2Web websites, same seed.

> **Setup:** Qwen 3.5-4B, LoRA rank 32. RL trained with `train_simple.py` (pure SSIM reward, BS=16, GS=2, 8K tokens). Eval on original website screenshots.

**Rentalcars** — RL produces blue gradient + search form; batch 20 adds background image
| Reference | Base (1 turn) | RL Batch 10 | RL Batch 20 |
|-----------|--------------|-------------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_07/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_07/turn1.png) | ![rl10](eval_output/m2w-simple-batch10/example_07/turn1.png) | ![rl20](eval_output/m2w-simple-batch20/example_07/turn1.png) |

**Resy** — Batch 20 adds styled city grid buttons
| Reference | Base (1 turn) | RL Batch 10 | RL Batch 20 |
|-----------|--------------|-------------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_00/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_00/turn1.png) | ![rl10](eval_output/m2w-simple-batch10/example_00/turn1.png) | ![rl20](eval_output/m2w-simple-batch20/example_00/turn1.png) |

**eBay** — Consistent teal hero + brand cards across checkpoints
| Reference | Base (1 turn) | RL Batch 10 | RL Batch 20 |
|-----------|--------------|-------------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_05/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_05/turn1.png) | ![rl10](eval_output/m2w-simple-batch10/example_05/turn1.png) | ![rl20](eval_output/m2w-simple-batch20/example_05/turn1.png) |

**IKEA** — Batch 20 generates product cards with pricing
| Reference | Base (1 turn) | RL Batch 10 | RL Batch 20 |
|-----------|--------------|-------------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_03/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_03/turn1.png) | ![rl10](eval_output/m2w-simple-batch10/example_03/turn1.png) | ![rl20](eval_output/m2w-simple-batch20/example_03/turn1.png) |

**Carnival** — Batch 20 produces countdown timer + room grid with branding colors
| Reference | Base (1 turn) | RL Batch 10 | RL Batch 20 |
|-----------|--------------|-------------|-------------|
| ![ref](eval_output/m2w-4b-base-1turn/example_06/ref-render.png) | ![base](eval_output/m2w-4b-base-1turn/example_06/turn1.png) | ![rl10](eval_output/m2w-simple-batch10/example_06/turn1.png) | ![rl20](eval_output/m2w-simple-batch20/example_06/turn1.png) |

| # | Website | Base | RL Batch 10 | RL Batch 20 |
|---|---------|------|------------|------------|
| 1 | Resy | 0.745 | 0.743 | 0.738 |
| 2 | FoxSports | 0.497 | 0.395 | 0.416 |
| 3 | UnderArmour | 0.533 | 0.429 | 0.414 |
| 4 | IKEA | 0.518 | 0.542 | 0.498 |
| 5 | Yelp | 0.471 | 0.474 | 0.366 |
| 6 | eBay | 0.716 | 0.627 | 0.655 |
| 7 | Carnival | 0.550 | 0.387 | 0.544 |
| 8 | Rentalcars | 0.396 | 0.573 | 0.436 |
| 9 | Viator | 0.322 | 0.328 | 0.319 |
| 10 | SoundCloud | 0.610 | 0.228 | 0.199 |
| **Avg SSIM** | | **0.536** | **0.472** | **0.459** |
| **Avg Reward** | | **-0.677** | **-0.698** | **-0.599** |

---

## 4. Task 3: Multi-Turn Analyze-Fix Agent (4B on Mind2Web)

Model generates HTML → sees target vs its output side-by-side → analyzes differences → fixes. Same 10 Mind2Web websites as Task 2.

### Base 4B (2 turns, analyze-fix)

> **Setup:** Same base 4B, 2 turns with analyze-fix step. No RL training — testing if the base model can self-correct.

**Resy** (SSIM 0.753)
| Reference | Base Turn 1 |
|-----------|------------|
| ![ref](eval_output/m2w-4b-base-2turns-v2/example_00/ref-render.png) | ![t1](eval_output/m2w-4b-base-2turns-v2/example_00/turn1.png) |

**eBay** (SSIM 0.630)
| Reference | Base Turn 1 |
|-----------|------------|
| ![ref](eval_output/m2w-4b-base-2turns-v2/example_05/ref-render.png) | ![t1](eval_output/m2w-4b-base-2turns-v2/example_05/turn1.png) |

**UnderArmour** (SSIM 0.470)
| Reference | Base Turn 1 |
|-----------|------------|
| ![ref](eval_output/m2w-4b-base-2turns-v2/example_02/ref-render.png) | ![t1](eval_output/m2w-4b-base-2turns-v2/example_02/turn1.png) |

**Rentalcars** (SSIM 0.528)
| Reference | Base Turn 1 |
|-----------|------------|
| ![ref](eval_output/m2w-4b-base-2turns-v2/example_07/ref-render.png) | ![t1](eval_output/m2w-4b-base-2turns-v2/example_07/turn1.png) |

| # | Website | Turn 1 SSIM | Turn 2 SSIM | Delta |
|---|---------|------------|------------|-------|
| 1 | Resy | 0.753 | 0.753 | +0.000 |
| 2 | FoxSports | 0.409 | 0.356 | -0.054 |
| 3 | UnderArmour | 0.470 | 0.468 | -0.002 |
| 4 | IKEA | 0.496 | 0.496 | +0.000 |
| 5 | Yelp | 0.324 | 0.324 | +0.000 |
| 6 | eBay | 0.630 | 0.628 | -0.002 |
| 7 | Carnival | 0.551 | 0.552 | +0.001 |
| 8 | Rentalcars | 0.528 | 0.545 | +0.017 |
| 9 | Viator | 0.352 | 0.352 | +0.000 |
| 10 | SoundCloud | 0.205 | 0.206 | +0.001 |
| **Avg** | | **0.472** | **0.468** | **-0.004** |

*Analyze-fix barely helps on base model — the model isn't trained to use visual feedback. RL training with `train_agent.py` in progress to improve this.*

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

### 4B Base Flow Eval (2 turns, analyze-fix)

> **Setup:** Qwen 3.5-4B base (no RL), 5 Mind2Web tasks with action sequences, 2 turns with analyze-fix. Model sees flow screenshots, generates interactive HTML/JS, we run actions via Playwright and compare per-step SSIM.

| Task | Website | Initial SSIM | Avg Step SSIM | Actions |
|------|---------|-------------|---------------|---------|
| 1 | sports.yahoo | 0.414 | 0.383 | 4 |
| 2 | underarmour | 0.472 | 0.528 | 16 |
| 3 | rottentomatoes | 0.628 | 0.643 | 5 |
| 4 | sports.yahoo | 0.174 | 0.180 | 3 |
| 5 | sports.yahoo | 0.378 | 0.373 | 2 |
| **Avg** | | **0.413** | **0.421** | |

Comparison with GPT-5.4:

| Model | Avg Initial SSIM | Avg Step SSIM |
|-------|-----------------|---------------|
| GPT-5.4 | **0.670** | **0.560** |
| 4B Base | 0.413 | 0.421 |

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
