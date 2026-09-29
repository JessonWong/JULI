"""Utility script for running inference on open-weight causal LLMs.
Provides fine-grained control over the sampling loop so you can inspect and
manipulate logits before every token is committed."""

# Copyright 2025-2026 The JULI Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modeling_biasnet import BiasNet
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with explicit control over sampling.")
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--tokenizer_name", default=None, help="Optional tokenizer identifier if it differs from the model.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--prompt", default=None, help="Single prompt for quick testing or streaming.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Multiple prompts supplied via the command line.")
    parser.add_argument("--prompt_file", default=None, help="Path to a text file with one prompt per line.")
    parser.add_argument("--system_prompt", default=None, help="Optional system prompt when using chat templates.")
    parser.add_argument("--use_chat_template", action="store_true", help="Apply the tokenizer chat template.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling. Set to 0 for greedy decode.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability mass. Set to 1.0 to disable.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling. Set to 0 to disable.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalise previously generated tokens (>1.0).")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Prevent repeating n-grams of this size. 0 disables it.")
    parser.add_argument("--min_length", type=int, default=0, help="Enforce a minimum generation length before EOS can appear.")
    parser.add_argument("--stop_sequence", action="append", default=None, help="Stop generation when this substring is produced. Repeatable.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", action="store_true", help="Stream decoded text for a single prompt.")
    parser.add_argument("--print_token_info", action="store_true", help="Print per-step token probabilities for active sequences.")
    parser.add_argument("--token_info_top_k", type=int, default=5, help="Number of tokens to display when printing step info.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of prompts to process together.")
    parser.add_argument("--device", default=None, help="Torch device specifier, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="Model weights dtype.")
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional device map for model parallelism, e.g. 'auto' or a JSON mapping of module names to devices.",
    )
    parser.add_argument("--output_dir", default=None, help="If set, write each generated completion to this directory.")
    parser.add_argument("--output_json", default=None, help="If set, append all generations into this JSONL file.")
    parser.add_argument("--biasnet_ckpt", default=None, help="Path to a BiasNet checkpoint. If unset, BiasNet is disabled.")
    return parser.parse_args()


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts:
        prompts.extend([p for p in args.prompts if p])
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if cleaned:
                    prompts.append(cleaned)
    if not prompts:
        raise ValueError("No prompts were provided. Use --prompt, --prompts, or --prompt_file.")
    return prompts


def prepare_prompt(raw_prompt: str, tokenizer, use_chat_template: bool, system_prompt: Optional[str]) -> str:
    if not use_chat_template:
        return raw_prompt
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    conversation.append({"role": "user", "content": raw_prompt})
    return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(seq), size):
        yield seq[index : index + size]


def load_model_and_tokenizer(args: argparse.Namespace):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    tokenizer_name = args.tokenizer_name or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = None
    if args.device_map:
        candidate = args.device_map.strip()
        potential_path = Path(candidate)
        if potential_path.is_file():
            with potential_path.open("r", encoding="utf-8") as handle:
                device_map = json.load(handle)
        else:
            try:
                device_map = json.loads(candidate)
            except json.JSONDecodeError:
                device_map = candidate
    if device_map is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            device_map=device_map,
        )

        def _resolve_primary_device(model_obj) -> torch.device:
            def _coerce_device(value):
                if isinstance(value, torch.device):
                    return value
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
                if isinstance(value, str):
                    if value.lower() == "disk":
                        return None
                    return torch.device(value)
                return None

            if hasattr(model_obj, "hf_device_map") and model_obj.hf_device_map:
                preferred_order = [
                    "model.embed_tokens",
                    "transformer.embed_tokens",
                    "transformer.wte",
                    "gpt_neox.embed_in",
                    "lm_head",
                ]
                for module_name in preferred_order:
                    if module_name in model_obj.hf_device_map:
                        device_candidate = _coerce_device(model_obj.hf_device_map[module_name])
                        if device_candidate is not None:
                            return device_candidate
                for mapped_device in model_obj.hf_device_map.values():
                    device_candidate = _coerce_device(mapped_device)
                    if device_candidate is not None:
                        return device_candidate
            try:
                return next(model_obj.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        device = _resolve_primary_device(model)
    else:
        device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
        ).to(device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer, device


def load_biasnet(ckpt_path: Optional[str], device: torch.device) -> Optional[BiasNet]:
    if not ckpt_path:
        return None
    bias_model = BiasNet.from_pretrained(ckpt_path)
    bias_model = bias_model.to(device)
    bias_model.set_up_proj()
    bias_model.eval()
    return bias_model


def trim_stop_sequences(text: str, stop_sequences: Optional[List[str]]) -> str:
    if not stop_sequences:
        return text
    earliest: Optional[int] = None
    for stop in stop_sequences:
        if not stop:
            continue
        index = text.find(stop)
        if index != -1 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text
    return text[:earliest]


def log_token_info(step: int, probs: torch.Tensor, tokenizer, args: argparse.Namespace, finished_mask: torch.Tensor) -> None:
    top_k = min(args.token_info_top_k, probs.size(-1))
    probs_cpu = probs.detach().cpu()
    finished_cpu = finished_mask.detach().cpu()
    for batch_idx in range(probs_cpu.size(0)):
        if finished_cpu[batch_idx]:
            continue
        values, indices = torch.topk(probs_cpu[batch_idx], k=top_k)
        tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
        pieces = [f"{repr(token)}:{value:.4f}" for token, value in zip(tokens, values.tolist())]
        print(f"[step {step} batch {batch_idx}] " + ", ".join(pieces))


@torch.no_grad()
def manual_generate(
    model,
    tokenizer,
    prompts: List[str],
    args: argparse.Namespace,
    device: torch.device,
    bias_model: Optional[BiasNet],
    stream: bool = False,
) -> Optional[List[str]]:
    prepared_prompts = [prepare_prompt(p, tokenizer, args.use_chat_template, args.system_prompt) for p in prompts]
    batch_encoding = tokenizer(prepared_prompts, return_tensors="pt", padding=True)
    prompt_ids = batch_encoding["input_ids"].to(device)
    attention_mask = batch_encoding["attention_mask"].to(device)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define a pad_token or pad_token_id for manual generation.")

    batch_size = prompt_ids.size(0)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
    decoded_cache = ["" for _ in range(batch_size)]
    stop_sequences = [s for s in (args.stop_sequence or []) if s]

    generated_ids = prompt_ids
    generated_attention = attention_mask
    past_key_values = None
    next_input_ids = prompt_ids
    supports_cache: Optional[bool] = None

    for step in range(args.max_new_tokens):
        if supports_cache is False:
            model_input_ids = generated_ids
        else:
            model_input_ids = next_input_ids
        model_kwargs = {
            "input_ids": model_input_ids,
            "attention_mask": generated_attention,
            "use_cache": True,
        }
        if supports_cache:
            model_kwargs["past_key_values"] = past_key_values
        outputs = model(**model_kwargs)
        if supports_cache is None:
            supports_cache = outputs.past_key_values is not None
        if supports_cache:
            past_key_values = outputs.past_key_values
        else:
            past_key_values = None

        logits_full = outputs.logits.float()
        if supports_cache is False:
            positions = generated_attention.sum(dim=1) - 1
            logits = logits_full[torch.arange(batch_size, device=device), positions]
        elif step == 0:
            prompt_positions = attention_mask.sum(dim=1) - 1
            logits = logits_full[torch.arange(batch_size, device=device), prompt_positions]
        else:
            logits = logits_full[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        if bias_model is not None:
            bias_input = log_probs
            log_probs = log_probs + bias_model(bias_input)
        probs = torch.exp(log_probs)

        if args.temperature is not None and args.temperature <= 0.0:
            next_tokens = torch.argmax(log_probs, dim=-1)
        else:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        next_tokens = next_tokens.masked_fill(finished, pad_token_id)

        if args.print_token_info:
            log_token_info(step, probs, tokenizer, args, finished)

        for idx in range(batch_size):
            if finished[idx]:
                continue
            token_id = int(next_tokens[idx])
            generated_tokens[idx].append(token_id)
            new_text = tokenizer.decode(generated_tokens[idx], skip_special_tokens=True)
            if stream and idx == 0:
                delta = new_text[len(decoded_cache[idx]) :]
                if delta:
                    print(delta, end="", flush=True)
            decoded_cache[idx] = new_text

        if eos_token_id is not None:
            eos_mask = (next_tokens == eos_token_id) & (~finished)
            finished |= eos_mask

        if stop_sequences:
            for idx, text in enumerate(decoded_cache):
                if finished[idx]:
                    continue
                trimmed = trim_stop_sequences(text, stop_sequences)
                if len(trimmed) < len(text):
                    finished[idx] = True
                    decoded_cache[idx] = trimmed
                    if trimmed:
                        tokenized = tokenizer(trimmed, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                        generated_tokens[idx] = tokenized.tolist()
                    else:
                        generated_tokens[idx] = []

        next_input_ids = next_tokens.unsqueeze(-1)
        generated_ids = torch.cat([generated_ids, next_input_ids], dim=-1)
        attention_extension = (~finished).unsqueeze(-1).to(generated_attention.dtype)
        generated_attention = torch.cat([generated_attention, attention_extension], dim=-1)

        if finished.all():
            break

    completions: List[str] = []
    for tokens, _ in zip(generated_tokens, decoded_cache):
        clean_tokens = [tok for tok in tokens if tok != pad_token_id]
        if eos_token_id is not None:
            while clean_tokens and clean_tokens[-1] == eos_token_id:
                clean_tokens.pop()
        text = tokenizer.decode(clean_tokens, skip_special_tokens=True)
        text = trim_stop_sequences(text, stop_sequences)
        completions.append(text.strip())
    if stream:
        print()
    return completions


def write_outputs(
    output_dir: Path,
    prompts: List[str],
    completions: List[str],
    start_index: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = start_index
    for prompt, completion in zip(prompts, completions):
        filename = output_dir / f"sample_{index:05d}.txt"
        with filename.open("w", encoding="utf-8") as handle:
            handle.write("PROMPT:\n")
            handle.write(prompt)
            handle.write("\n\nCOMPLETION:\n")
            handle.write(completion)
            handle.write("\n")
        index += 1
    return index


def append_jsonl(path: Path, prompts: List[str], completions: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for prompt, completion in zip(prompts, completions):
            record = {"prompt": prompt, "completion": completion}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    try:
        prompts = collect_prompts(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    if args.stream and len(prompts) != 1:
        print("Streaming mode requires exactly one prompt.", file=sys.stderr)
        sys.exit(1)
    set_seed(args.seed)
    model, tokenizer, device = load_model_and_tokenizer(args)
    bias_model = load_biasnet(args.biasnet_ckpt, device)

    output_dir = Path(args.output_dir) if args.output_dir else None
    json_path = Path(args.output_json) if args.output_json else None
    next_file_index = 0

    if args.stream:
        completions = manual_generate(model, tokenizer, prompts, args, device, bias_model, stream=True)
        if output_dir:
            next_file_index = write_outputs(output_dir, prompts, completions, next_file_index)
        if json_path:
            append_jsonl(json_path, prompts, completions)
        return

    for batch in chunked(prompts, args.batch_size):
        completions = manual_generate(model, tokenizer, batch, args, device, bias_model, stream=False)
        for prompt, completion in zip(batch, completions):
            print("=== Prompt ===")
            print(prompt)
            print("=== Completion ===")
            print(completion)
            print()
        if output_dir:
            next_file_index = write_outputs(output_dir, batch, completions, next_file_index)
        if json_path:
            append_jsonl(json_path, batch, completions)

if __name__ == "__main__":
    main()
