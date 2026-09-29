# Copyright 2025-2026 The JULI Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache target-token log probabilities from Gemini through Vertex AI."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from google import genai
from google.genai.types import GenerateContentConfig
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Gemini top-k log probabilities for prompt/answer pairs."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Local JSON or JSONL file containing prompt/answer records.",
    )
    parser.add_argument(
        "--input_format",
        choices=["auto", "json", "jsonl"],
        default="auto",
    )
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--answer_field", default="answer")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--model_name",
        "--model_name_or_path",
        dest="model_name",
        default="gemini-2.5-pro",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="google/gemma-3-1b-pt",
        help="Tokenizer/config whose token IDs and vocabulary match the API response.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
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
    parser.add_argument("--logprobs", type=int, default=5)
    parser.add_argument("--api_max_output_tokens", type=int, default=5)
    parser.add_argument("--request_delay", type=float, default=1.0)
    return parser.parse_args()


def load_records(path: Path, input_format: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if input_format == "auto":
        input_format = "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "json"

    if input_format == "jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Record on line {line_number} is not a JSON object.")
                records.append(record)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("JSON input must be an object or a list of objects.")
    return payload


def select_examples(records: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    if args.logprobs <= 0 or args.api_max_output_tokens <= 0:
        raise ValueError("--logprobs and --api_max_output_tokens must be positive.")
    if args.request_delay < 0:
        raise ValueError("--request_delay must be non-negative.")

    selected = records[args.start_index : args.start_index + args.max_samples]
    examples: List[Dict[str, str]] = []
    for offset, record in enumerate(selected, start=args.start_index):
        prompt = record.get(args.prompt_field)
        answer = record.get(args.answer_field)
        if not isinstance(prompt, str) or not isinstance(answer, str):
            raise ValueError(
                f"Record {offset} must contain string fields "
                f"{args.prompt_field!r} and {args.answer_field!r}."
            )
        if not prompt or not answer:
            raise ValueError(f"Record {offset} contains an empty prompt or answer.")
        examples.append({"prompt": prompt, "answer": answer})
    return examples


def format_gemini_prompt(question: str, answer_prefix: Optional[str] = None):
    contents = [{"role": "user", "parts": [{"text": question}]}]
    if answer_prefix:
        contents.append({"role": "model", "parts": [{"text": answer_prefix}]})
    return contents


def get_logprobs_gemini(
    client,
    tokenizer,
    vocab_size: int,
    question: str,
    answer: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, torch.Tensor]]:
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError("The answer produced no tokenizer tokens.")

    rows = []
    for index in range(len(answer_ids)):
        prefix = tokenizer.decode(answer_ids[:index]) if index else None
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
            print("Skipping a sample because Gemini returned no candidate.")
            return None
        logprobs_result = completion.candidates[0].logprobs_result
        if not logprobs_result or not logprobs_result.top_candidates:
            print("Skipping a sample because Gemini returned no log probabilities.")
            return None
        candidates = logprobs_result.top_candidates[0].candidates
        if not candidates:
            print("Skipping a sample because Gemini returned an empty candidate list.")
            return None

        fill_value = min(item.log_probability for item in candidates) - 10.0
        row = torch.full((vocab_size,), fill_value, dtype=torch.float32)
        for item in candidates:
            token_id = getattr(item, "token_id", None)
            if token_id is None:
                token = getattr(item, "token", None)
                token_id = tokenizer.convert_tokens_to_ids(token) if token is not None else None
            if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                row[token_id] = item.log_probability
        rows.append(row.unsqueeze(0))
        if args.request_delay:
            time.sleep(args.request_delay)

    log_probs = torch.cat(rows, dim=0).unsqueeze(0)
    labels = torch.tensor(answer_ids, dtype=torch.long).unsqueeze(0)
    return {"log_probs": log_probs, "labels": labels}


def cache_key(prompt: str, answer: str) -> str:
    serialized = json.dumps([prompt, answer], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if not args.vertex_project:
        raise ValueError(
            "Provide --vertex_project or set the GOOGLE_CLOUD_PROJECT environment variable."
        )
    examples = select_examples(load_records(args.input_file, args.input_format), args)
    if not examples:
        raise ValueError("No records were selected from the input file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    vocab_size = AutoConfig.from_pretrained(args.tokenizer_name).vocab_size
    client = genai.Client(
        vertexai=True,
        project=args.vertex_project,
        location=args.vertex_location,
    )

    saved = 0
    skipped = 0
    failed = 0
    for example in tqdm(examples, desc="Caching logits"):
        output_path = args.output_dir / f"{cache_key(example['prompt'], example['answer'])}.pt"
        if output_path.exists():
            skipped += 1
            continue
        result = get_logprobs_gemini(
            client,
            tokenizer,
            vocab_size,
            example["prompt"],
            example["answer"],
            args,
        )
        if result is None:
            failed += 1
            continue
        torch.save(result, output_path)
        saved += 1

    print(
        f"Selected {len(examples)} samples; saved {saved}, skipped {skipped}, failed {failed} "
        f"in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
