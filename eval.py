"""
Quick eval: sample from a trained model on held-out screenshots and visually inspect.

Usage:
    python eval.py --model_path <tinker-model-path> --n 5
"""

import argparse
import json
import os
import random

import tinker
from tinker import types
from PIL import Image

from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import AutoImageProcessor
from reward import extract_html_from_response

MODEL = "Qwen/Qwen3.5-4B"
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "data", "manifest.json")

SYSTEM_PROMPT = (
    "You are an expert at converting screenshots of web pages into HTML/CSS code. "
    "Given a screenshot, output ONLY the inner HTML snippet (no <html>, <head>, or <body> tags) "
    "that reproduces the visual appearance. Wrap your code in ```html ... ```."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None,
                        help="Tinker model path for trained weights. If None, uses base model.")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to evaluate")
    args = parser.parse_args()

    with open(MANIFEST_PATH) as f:
        dataset = json.load(f)

    samples = random.sample(dataset, min(args.n, len(dataset)))

    tokenizer = get_tokenizer(MODEL)
    renderer_name = "qwen3_5_disable_thinking"
    image_processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
    renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor=image_processor)

    service_client = tinker.ServiceClient()

    if args.model_path:
        sampling_client = service_client.create_sampling_client(model_path=args.model_path)
    else:
        # Use base model for comparison
        training_client = service_client.create_lora_training_client(
            base_model=MODEL, rank=32,
        )
        sampling_client = training_client.save_weights_and_get_sampling_client()

    sampling_params = types.SamplingParams(
        max_tokens=512,
        stop=renderer.get_stop_sequences(),
        temperature=0.3,
    )

    out_dir = os.path.join(os.path.dirname(__file__), "eval_output")
    os.makedirs(out_dir, exist_ok=True)

    for i, item in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Example {i+1}: {item['screenshot']}")
        print(f"Ground truth HTML:\n{item['html']}")

        img = Image.open(item["screenshot"]).convert("RGB")
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Generate the HTML/CSS snippet that reproduces this screenshot."},
                ],
            },
        ]
        prompt = renderer.build_generation_prompt(convo)

        result = sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        parsed_msg, _ = renderer.parse_response(result.sequences[0].tokens)
        content = get_text_content(parsed_msg)
        generated_html = extract_html_from_response(content)

        print(f"\nGenerated HTML:\n{generated_html}")

        # Save side-by-side
        with open(os.path.join(out_dir, f"example_{i}.json"), "w") as f:
            json.dump({
                "screenshot": item["screenshot"],
                "ground_truth": item["html"],
                "generated": generated_html,
                "raw_response": content,
            }, f, indent=2)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
