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

### 4B Base Model on Real Websites

**Resy (SSIM 0.816)** — best result
| Reference | Generated |
|-----------|-----------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_00/ref.png) | ![gen](eval_output/single_image_rl/4b-base-simple/example_00/gen.png) |

**eBay (SSIM 0.727)**
| Reference | Generated |
|-----------|-----------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_05/ref.png) | ![gen](eval_output/single_image_rl/4b-base-simple/example_05/gen.png) |

**IKEA (SSIM 0.580)**
| Reference | Generated |
|-----------|-----------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_03/ref.png) | ![gen](eval_output/single_image_rl/4b-base-simple/example_03/gen.png) |

**SoundCloud (SSIM 0.206)** — hardest
| Reference | Generated |
|-----------|-----------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_09/ref.png) | ![gen](eval_output/single_image_rl/4b-base-simple/example_09/gen.png) |

### After 10 Batches RL Training

**SoundCloud: 0.206 → 0.545 (+0.339!)** — biggest improvement
| Reference | Base (SSIM 0.206) | RL Batch 10 (SSIM 0.545) |
|-----------|-------------------|--------------------------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_09/ref.png) | ![base](eval_output/single_image_rl/4b-base-simple/example_09/gen.png) | ![rl](eval_output/single_image_rl/4b-simple-batch10/example_09/gen.png) |

**Resy: 0.816 → 0.849 (+0.033)** — already good, got better
| Reference | Base (SSIM 0.816) | RL Batch 10 (SSIM 0.849) |
|-----------|-------------------|--------------------------|
| ![ref](eval_output/single_image_rl/4b-base-simple/example_00/ref.png) | ![base](eval_output/single_image_rl/4b-base-simple/example_00/gen.png) | ![rl](eval_output/single_image_rl/4b-simple-batch10/example_00/gen.png) |

---

## 3. Task 2: Multi-Turn Analyze-Fix Agent

Model generates HTML → we render + create red diff → model analyzes what's wrong → model fixes.

### The Visual Diff Feedback Loop

| Target | Model's Output | Diff (red = wrong) |
|--------|---------------|---------------------|
| ![ref](eval_output/frontier_baselines/agent_demo_v2/reference.png) | ![turn1](eval_output/frontier_baselines/agent_demo_v2/turn1.png) | ![diff1](eval_output/frontier_baselines/agent_demo_v2/diff1.png) |

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

### GPT-5.4 on Budget Car Rental (16 action steps!)

| Step | Action | SSIM | Screenshot |
|------|--------|------|------------|
| Initial | Page load | 0.764 | ![step0](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_00/step_0_gen.png) |
| Reference for initial: | | | ![ref0](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_00/step_0_ref.png) |

### GPT-5.4 on SpotHero Parking (8 action steps)

| Step | Action | SSIM | Screenshot |
|------|--------|------|------------|
| Initial | Page load | 0.865 | ![step0](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_01/step_0_gen.png) |
| Reference: | | | ![ref0](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_01/step_0_ref.png) |

### GPT-5.4 on Resy (3 action steps)

| Step | Action | SSIM | Screenshot |
|------|--------|------|------------|
| Initial | Page load | 0.817 | ![step0](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_04/step_0_gen.png) |
| Reference: | | | ![ref0](eval_output/flow_interactive/mind2web-gpt54-flow-analyze-v2/task_04/step_0_ref.png) |

---

## 5. Long Training Runs

### Experiment 9: 27B on WebSight + D2C (Best Early Result)
90 batches, reward climbed steadily. **+0.231 improvement, 9/10 wins.**

| Example | Base | RL | Delta |
|---------|------|-----|-------|
| ![ref](eval_output/single_image_rl/exp9-batch75/example_02/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_02/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_02/rl.png) | +0.199 |
| ![ref](eval_output/single_image_rl/exp9-batch75/example_06/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_06/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_06/rl.png) | +0.749 |
| ![ref](eval_output/single_image_rl/exp9-batch75/example_09/ref-render.png) | ![base](eval_output/single_image_rl/exp9-batch75/example_09/base.png) | ![rl](eval_output/single_image_rl/exp9-batch75/example_09/rl.png) | +0.469 |

### Current Run: 4B on Mind2Web (Real Websites)
One-shot, pure SSIM reward, 500 Mind2Web landing pages. Training live.

```
Batch 0:  reward=0.118
Batch 5:  reward=-0.063  (checkpoint saved)
Batch 10: reward=-0.123  (checkpoint saved)
Batch 12: reward=0.062   (climbing back)
```

---

## 6. Frontier Model Comparisons

### GPT-5.4 on Hard Design2Code
Avg reward 0.716, SSIM 0.906. Generates 1-4K chars for 100K+ char source pages.

| Reference | GPT-5.4 Generated | SSIM |
|-----------|--------------------|------|
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
| Playwright selectors=None (74% of clicks failed) | Text-based matching |
| Mind2Web raw_html has no CSS (DOM dump) | Compare against original screenshots |
| 27B multi-image: 4.5 hrs/batch | One-shot single image, parallel sampling |
| HTML cutoff (white screens) | 16K token limit |
| Viewport sizes causing bad comparisons | 1024x768 or 1280x720 standard |
