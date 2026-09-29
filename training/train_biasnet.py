# Copyright 2025-2026 The JULI Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModelForCausalLM

import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from modeling_biasnet import BiasConfig, BiasNet

class CachedLogitsDataset(Dataset):
    """Dataset backed by cached logits generated via pre_logits_openweight.py."""

    def __init__(
        self,
        data_dir: str,
        dtype: torch.dtype = torch.float32,
        max_tokens_per_sample: Optional[int] = None,
    ) -> None:
        super().__init__()
        file_paths: List[str] = sorted(
            [
                os.path.join(data_dir, fname)
                for fname in os.listdir(data_dir)
                if fname.endswith(".pt")
            ]
        )
        if not file_paths:
            raise FileNotFoundError(f"No .pt files found in {data_dir}.")

        if max_tokens_per_sample is not None and max_tokens_per_sample <= 0:
            raise ValueError("--sample_length must be a positive integer when provided.")

        self.dtype = dtype
        logits_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        vocab_size: Optional[int] = None

        for path in file_paths:
            payload = torch.load(path, map_location="cpu")
            log_probs = payload["log_probs"]
            labels = payload["labels"]
            if log_probs.dim() != 3 or labels.dim() != 2:
                raise ValueError(
                    "Expected log_probs of shape [1, seq_len, vocab] and labels [1, seq_len]."
                )
            if log_probs.shape[0] != 1:
                raise ValueError("Only single-sample batches are supported in cached files.")
            if vocab_size is None:
                vocab_size = log_probs.shape[2]
            logit_slice = log_probs[0]
            label_slice = labels[0]
            if max_tokens_per_sample is not None:
                slice_len = min(max_tokens_per_sample, logit_slice.shape[0])
                if slice_len == 0:
                    continue
                logit_slice = logit_slice[:slice_len]
                label_slice = label_slice[:slice_len]
            logits_chunks.append(logit_slice.to(dtype))
            label_chunks.append(label_slice.long())

        assert vocab_size is not None
        if not logits_chunks:
            raise ValueError(
                "No tokens available after applying --sample_length. "
                "Increase the value or remove the restriction."
            )
        self.vocab_size = vocab_size
        self.logits = torch.cat(logits_chunks, dim=0).contiguous()
        self.labels = torch.cat(label_chunks, dim=0).contiguous()
        self.num_tokens = self.labels.size(0)

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= self.num_tokens:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_tokens}.")
        logits = self.logits[index]
        label = self.labels[index]
        return logits, label



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiasNet on cached logits.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing cached .pt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store the trained BiasNet checkpoint.")
    parser.add_argument("--base_model_name_or_path", type=str, default=None, help="Hugging Face model id or local path used to derive hidden/vocab sizes.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom code supplied by the selected model repository.",
    )
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden size override. Required if base model config is not supplied.")
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size override. Defaults to cached logits vocab size.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--sample_length",
        type=int,
        default=None,
        help="Limit training to the first N tokens from each cached sample (e.g., 20).",
    )
    parser.add_argument(
        "--init_lm_head",
        action="store_true",
        help="Initialise BiasNet.lm_head from the base model weights (deprecated; prefer --lm_head_init).",
    )
    parser.add_argument(
        "--lm_head_init",
        choices=["none", "copy", "optimize", "data_aware", "zeros", "ones"],
        default=None,
        help="Strategy for initialising BiasNet.lm_head. Defaults to 'copy' when --init_lm_head is used, otherwise 'none'.",
    )
    parser.add_argument(
        "--lm_head_optimize_lr",
        type=float,
        default=1e-5,
        help="Learning rate used when --lm_head_init=optimize.",
    )
    parser.add_argument(
        "--lm_head_optimize_batch_size",
        type=int,
        default=1024,
        help="Batch size used when --lm_head_init=optimize.",
    )
    parser.add_argument(
        "--lm_head_optimize_epochs",
        type=int,
        default=1,
        help="Number of epochs for the optimization-based lm_head initialiser.",
    )
    parser.add_argument(
        "--lm_head_optimize_init",
        choices=["random", "zeros", "ones"],
        default="random",
        help="Initial weight pattern used before optimisation when --lm_head_init=optimize.",
    )
    parser.add_argument(
        "--lm_head_data_sample_size",
        type=int,
        default=2048,
        help="Number of cached tokens sampled when --lm_head_init=data_aware.",
    )
    parser.add_argument(
        "--lm_head_data_scale",
        choices=["singular", "normalize", "none"],
        default="singular",
        help="Row scaling applied to PCA components for --lm_head_init=data_aware.",
    )
    parser.add_argument("--mixed_precision", action="store_true", help="Use torch.autocast for mixed precision on CUDA.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimize_lm_head_weights(
    vocab_size: int,
    hidden_size: int,
    device: torch.device,
    lr: float,
    batch_size: int,
    epochs: int,
    init_mode: str,
) -> torch.Tensor:
    """Optimise a vocab-sized weight matrix to minimise pairwise cosine similarity."""

    if init_mode == "zeros":
        init_tensor = torch.zeros(vocab_size, hidden_size, device=device, dtype=torch.float32)
    elif init_mode == "ones":
        init_tensor = torch.ones(vocab_size, hidden_size, device=device, dtype=torch.float32)
    elif init_mode == "random":
        init_tensor = torch.randn(vocab_size, hidden_size, device=device, dtype=torch.float32) * 0.01
    else:
        raise ValueError(f"Unsupported lm_head optimisation init mode: {init_mode}")

    weight = nn.Parameter(init_tensor)
    optimizer = torch.optim.Adam([weight], lr=lr)
    total_steps = (vocab_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        permutation = torch.randperm(vocab_size, device=device)
        for step in range(total_steps):
            start = step * batch_size
            end = min(start + batch_size, vocab_size)
            indices = permutation[start:end]

            optimizer.zero_grad(set_to_none=True)
            batch_vectors = weight[indices]
            normalized = F.normalize(batch_vectors, p=2, dim=1)
            similarity = normalized @ normalized.t()
            similarity.fill_diagonal_(0)
            loss = similarity.pow(2).mean()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        weight.copy_(F.normalize(weight, p=2, dim=1, eps=1e-12))

    return weight.detach().cpu()


def data_aware_lm_head_weights(
    dataset: CachedLogitsDataset,
    hidden_size: int,
    sample_size: int,
    seed: int,
    scale_mode: str,
) -> torch.Tensor:
    """Initialise lm_head using principal components estimated from cached logits."""

    total_tokens = dataset.num_tokens
    if total_tokens == 0:
        raise ValueError("CachedLogitsDataset is empty; cannot derive data-aware lm_head.")
    if sample_size < 2:
        raise ValueError("--lm_head_data_sample_size must be at least 2.")

    sample_size = min(sample_size, total_tokens)
    logits = dataset.logits
    device = logits.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sample_indices = torch.randperm(total_tokens, generator=generator, device=device)[:sample_size]
    sample = logits.index_select(0, sample_indices).to(torch.float32)
    sample = sample - sample.mean(dim=0, keepdim=True)

    # return sample.transpose(0,1).contiguous().cpu()
    target_rank = min(hidden_size, sample_size - 1, sample.shape[1])
    print("hidden_size:", hidden_size)
    print("target_rank:", target_rank)
    if target_rank <= 0:
        raise ValueError(
            f"Unable to compute PCA with sample_size={sample_size}. "
            "Increase --lm_head_data_sample_size or provide more cached logits."
        )

    with torch.no_grad():
        _, singular_values, right_vecs = torch.pca_lowrank(sample, q=target_rank, center=False)
        components = right_vecs[:, :target_rank]

        if scale_mode == "singular":
            components = components * singular_values[:target_rank].unsqueeze(0)
        elif scale_mode == "normalize":
            components = F.normalize(components, p=2, dim=1)

        if target_rank < hidden_size:
            remainder = hidden_size - target_rank
            random_block = torch.randn(
                components.size(0),
                remainder,
                generator=generator,
                device=components.device,
                dtype=components.dtype,
            )
            # Orthonormalise the random block to avoid duplicating directions.
            random_block, _ = torch.linalg.qr(random_block, mode="reduced")
            components = torch.cat([components, random_block[:, :remainder]], dim=1)

    return components[:, :hidden_size].contiguous().cpu()


def prepare_biasnet(
    args: argparse.Namespace, dataset: CachedLogitsDataset, device: torch.device
) -> BiasNet:
    if args.base_model_name_or_path:
        base_config = AutoConfig.from_pretrained(
            args.base_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if args.hidden_size is None:
            hidden_size = base_config.hidden_size
        else:
            hidden_size = args.hidden_size
        vocab_size = base_config.vocab_size
    else:
        if args.hidden_size is None:
            raise ValueError("hidden_size must be provided when base_model_name_or_path is not supplied.")
        hidden_size = args.hidden_size
        vocab_size = args.vocab_size or dataset.vocab_size

    config = BiasConfig(hidden_size=hidden_size, vocab_size=vocab_size)
    model = BiasNet(config)

    init_mode = args.lm_head_init or ("copy" if args.init_lm_head else "none")

    if init_mode == "copy":
        if not args.base_model_name_or_path:
            raise ValueError("--lm_head_init=copy requires --base_model_name_or_path to be set.")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        with torch.no_grad():
            model.lm_head.weight.copy_(base_model.lm_head.weight.float())
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif init_mode == "optimize":
        optim_device = device if device.type == "cuda" else torch.device("cpu")
        print(
            f"Initialising BiasNet lm_head with optimisation on {optim_device}"
            f" (epochs={args.lm_head_optimize_epochs}, batch_size={args.lm_head_optimize_batch_size}, lr={args.lm_head_optimize_lr})."
        )
        optimized_weights = optimize_lm_head_weights(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            device=optim_device,
            lr=args.lm_head_optimize_lr,
            batch_size=args.lm_head_optimize_batch_size,
            epochs=args.lm_head_optimize_epochs,
            init_mode=args.lm_head_optimize_init,
        )
        with torch.no_grad():
            model.lm_head.weight.copy_(optimized_weights.to(model.lm_head.weight.dtype))
    elif init_mode == "data_aware":
        print(
            f"Initialising BiasNet lm_head with data-aware PCA over {args.lm_head_data_sample_size} logits."
        )
        data_weights = data_aware_lm_head_weights(
            dataset=dataset,
            hidden_size=hidden_size,
            sample_size=args.lm_head_data_sample_size,
            seed=args.seed,
            scale_mode=args.lm_head_data_scale,
        )
        with torch.no_grad():
            model.lm_head.weight.copy_(data_weights.to(model.lm_head.weight.dtype))
    elif init_mode == "zeros":
        with torch.no_grad():
            model.lm_head.weight.zero_()
    elif init_mode == "ones":
        with torch.no_grad():
            model.lm_head.weight.fill_(1.0)
    elif init_mode != "none":
        raise ValueError(f"Unknown lm_head initialisation mode: {init_mode}")

    model.train()
    model.set_up_proj()
    for param in model.lm_head.parameters():
        param.requires_grad = False
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.lm_head_init is None:
        args.lm_head_init = "copy" if args.init_lm_head else "none"

    dataset = CachedLogitsDataset(
        args.data_dir,
        max_tokens_per_sample=args.sample_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = prepare_biasnet(args, dataset, device).to(device)
    if hasattr(model, "up_proj"):
        model.up_proj = model.up_proj.to(device)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)

    # model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(dataloader)
        for step, batch in enumerate(dataloader):
            logits, labels = batch
            logits = logits.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=args.mixed_precision and device.type == "cuda"):
                outputs = model(logits)
                outputs = outputs + logits
                loss = criterion(outputs, labels)
                loss_to_log = loss.detach()
                loss = loss / args.grad_accumulation

            scaler.scale(loss).backward()

            should_step = (step + 1) % args.grad_accumulation == 0 or (step + 1) == num_batches
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += loss_to_log.item() * labels.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    model.save_pretrained(args.output_dir)
    print(f"Saved BiasNet checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
