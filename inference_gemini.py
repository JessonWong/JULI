# Copyright 2025-2026 The JULI Authors
# SPDX-License-Identifier: Apache-2.0

"""Run JULI-controlled token generation against Gemini through Vertex AI."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from google import genai
from google.genai.types import GenerateContentConfig
from transformers import AutoTokenizer

from modeling_biasnet import BiasNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Gemini completions while applying a trained BiasNet at each step."
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to process. Repeat this option to supply multiple prompts.",
    )
    parser.add_argument(
        "--prompt_file",
        type=Path,
        default=None,
        help="Local UTF-8 text file with one prompt per non-empty line.",
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model_name", default="gemini-2.5-pro")
    parser.add_argument("--biasnet_ckpt", type=Path, required=True)
    parser.add_argument("--tokenizer_name", default="google/gemma-3-1b-pt")
    parser.add_argument(
        "--vertex_project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project; defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--vertex_location",
        default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        help="Vertex AI location; defaults to GOOGLE_CLOUD_LOCATION or 'global'.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--api_max_output_tokens", type=int, default=5)
    parser.add_argument("--request_delay", type=float, default=1.0)
    return parser.parse_args()


def collect_prompts(args: argparse.Namespace) -> List[str]:
    prompts = [prompt for prompt in (args.prompt or []) if prompt]
    if args.prompt_file:
        if not args.prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {args.prompt_file}")
        with args.prompt_file.open("r", encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    if not prompts:
        raise ValueError("Provide at least one --prompt or a non-empty --prompt_file.")
    return prompts


def format_gemini_prompt(question: str, answer_prefix: Optional[str] = None):
    contents = [{"role": "user", "parts": [{"text": question}]}]
    if answer_prefix:
        contents.append({"role": "model", "parts": [{"text": answer_prefix}]})
    return contents


def sample_next(log_probs: torch.Tensor, temperature: float) -> int:
    if temperature < 0:
        raise ValueError("--temperature must be non-negative.")
    if temperature == 0:
        return int(torch.argmax(log_probs, dim=-1).item())
    probabilities = torch.softmax(log_probs / temperature, dim=-1)
    return int(torch.multinomial(probabilities, 1).item())


def generate_with_juli(
    question: str,
    client,
    bias_net: BiasNet,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    generated_ids: List[int] = []
    vocab_size = int(bias_net.config.vocab_size)
    eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = set()
    elif isinstance(eos_token_ids, int):
        eos_token_ids = {eos_token_ids}
    else:
        eos_token_ids = set(eos_token_ids)

    for _ in range(args.max_new_tokens):
        prefix = tokenizer.decode(generated_ids, skip_special_tokens=False) if generated_ids else None
        completion = client.models.generate_content(
            model=args.model_name,
            contents=format_gemini_prompt(question, prefix),
            config=GenerateContentConfig(
                response_logprobs=True,
                logprobs=args.logprobs,
                max_output_tokens=args.api_max_output_tokens,
            ),
        )
        if not completion or not completion.candidates:
            break
        logprobs_result = completion.candidates[0].logprobs_result
        if not logprobs_result or not logprobs_result.top_candidates:
            break
        candidates = logprobs_result.top_candidates[0].candidates
        if not candidates:
            break

        fill_value = min(item.log_probability for item in candidates) - 10.0
        log_probs = torch.full(
            (1, vocab_size),
            fill_value,
            dtype=torch.float32,
            device=device,
        )
        for item in candidates:
            token_id = getattr(item, "token_id", None)
            if token_id is None:
                token = getattr(item, "token", None)
                token_id = tokenizer.convert_tokens_to_ids(token) if token is not None else None
            if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                log_probs[0, token_id] = item.log_probability

        with torch.no_grad():
            controlled_log_probs = log_probs + bias_net(log_probs)
        next_token_id = sample_next(controlled_log_probs, args.temperature)
        if next_token_id in eos_token_ids:
            break
        generated_ids.append(next_token_id)
        if args.request_delay:
            time.sleep(args.request_delay)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def validate_args(args: argparse.Namespace) -> None:
    if not args.vertex_project:
        raise ValueError(
            "Provide --vertex_project or set the GOOGLE_CLOUD_PROJECT environment variable."
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if args.logprobs <= 0 or args.api_max_output_tokens <= 0:
        raise ValueError("--logprobs and --api_max_output_tokens must be positive.")
    if args.request_delay < 0:
        raise ValueError("--request_delay must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    prompts = collect_prompts(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    bias_net = BiasNet.from_pretrained(args.biasnet_ckpt).to(device)
    bias_net.eval()
    bias_net.set_up_proj()

    tokenizer_vocab_size = int(tokenizer.vocab_size)
    biasnet_vocab_size = int(bias_net.config.vocab_size)
    if tokenizer_vocab_size != biasnet_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary ({tokenizer_vocab_size}) does not match "
            f"BiasNet vocabulary ({biasnet_vocab_size})."
        )

    client = genai.Client(
        vertexai=True,
        project=args.vertex_project,
        location=args.vertex_location,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    with args.output_json.open(mode, encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts, start=1):
            started = time.time()
            completion = generate_with_juli(
                prompt,
                client,
                bias_net,
                tokenizer,
                device,
                args,
            )
            handle.write(
                json.dumps(
                    {"prompt": prompt, "completion": completion},
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            print(f"Generated {index}/{len(prompts)} in {time.time() - started:.1f}s")

    print(f"Wrote {len(prompts)} records to {args.output_json}.")


if __name__ == "__main__":
    main()
