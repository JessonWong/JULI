# Copyright 2025-2026 The JULI Authors
# SPDX-License-Identifier: Apache-2.0

"""Cache target-token log probabilities from an open-weight causal LM."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache per-token log probabilities for prompt/answer pairs."
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
        help="Input format. 'auto' infers JSONL from .jsonl/.ndjson and JSON otherwise.",
    )
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--answer_field", default="answer")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--model_name_or_path",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model identifier or local model directory.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        help="Optional tokenizer identifier; defaults to --model_name_or_path.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Transformers device map, for example 'auto'.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom code supplied by the selected model repository.",
    )
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


def get_logprobs(model, tokenizer, question: str, answer: str) -> Dict[str, torch.Tensor]:
    question_template = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    whole_template = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    question_text = tokenizer.apply_chat_template(question_template, tokenize=False)
    whole_text = tokenizer.apply_chat_template(whole_template, tokenize=False)

    input_device = model.get_input_embeddings().weight.device
    input_ids = tokenizer(whole_text, return_tensors="pt").input_ids.to(input_device)
    question_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(input_device)

    special_token_ids = set(tokenizer.all_special_ids or [])
    while question_ids.shape[1] > 0 and question_ids[0, -1].item() in special_token_ids:
        question_ids = question_ids[:, :-1]

    compare_len = min(question_ids.shape[1], input_ids.shape[1])
    divergence_idx = compare_len
    for index in range(compare_len):
        if question_ids[0, index].item() != input_ids[0, index].item():
            divergence_idx = index
            break
    question_ids = question_ids[:, :divergence_idx]

    if question_ids.shape[1] == 0:
        raise ValueError("Unable to align prompt tokens with the full conversation.")
    if not torch.equal(question_ids, input_ids[:, : question_ids.shape[1]]):
        raise ValueError("Prompt tokens do not align with the beginning of the conversation.")

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    answer_ids = input_ids[:, question_ids.shape[1] :]
    answer_log_probs = log_probs[:, question_ids.shape[1] - 1 : input_ids.shape[1] - 1, :]
    if answer_ids.shape[1] != answer_log_probs.shape[1]:
        raise RuntimeError("Answer labels and log probabilities have different lengths.")
    return {"log_probs": answer_log_probs.cpu(), "labels": answer_ids.cpu()}


def cache_key(prompt: str, answer: str) -> str:
    serialized = json.dumps([prompt, answer], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    examples = select_examples(load_records(args.input_file, args.input_format), args)
    if not examples:
        raise ValueError("No records were selected from the input file.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    saved = 0
    skipped = 0
    for example in tqdm(examples, desc="Caching logits"):
        output_path = args.output_dir / f"{cache_key(example['prompt'], example['answer'])}.pt"
        if output_path.exists():
            skipped += 1
            continue
        result = get_logprobs(model, tokenizer, example["prompt"], example["answer"])
        torch.save(result, output_path)
        saved += 1

    print(
        f"Selected {len(examples)} samples; saved {saved}, skipped {skipped} existing files "
        f"in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
