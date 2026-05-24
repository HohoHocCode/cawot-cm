"""CLIP fine-tuning with symmetric InfoNCE loss.

V0 design choice: freeze backbone except last N transformer layers.
This gives a tractable, fast finetune on Colab T4 and is enough to show
that selection-quality affects downstream performance.

For V1+ switch to IRRA (BERT + ViT + SDM/IRR losses).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PABTrainDataset
from .embed import load_clip
from .utils import ensure_dir, get_device, setup_logger

logger = setup_logger("train")


def _unfreeze_last_n_layers(model, n: int) -> int:
    """Freeze all params, then unfreeze last `n` resblocks of visual + text + heads.

    Works for open_clip ViT-B/16. Returns number of trainable params.
    """
    for p in model.parameters():
        p.requires_grad = False

    # Visual transformer last n blocks
    if hasattr(model.visual, "transformer"):
        blocks = model.visual.transformer.resblocks
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
    # Visual final norm + projection
    for attr in ("ln_post", "proj"):
        m = getattr(model.visual, attr, None)
        if m is None:
            continue
        if hasattr(m, "parameters"):
            for p in m.parameters():
                p.requires_grad = True
        else:
            m.requires_grad = True

    # Text transformer last n blocks
    if hasattr(model, "transformer"):
        blocks = model.transformer.resblocks
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
    # Text final norm + projection
    for attr in ("ln_final",):
        m = getattr(model, attr, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True
    if hasattr(model, "text_projection"):
        model.text_projection.requires_grad = True

    # Logit scale always trainable
    if hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {n_trainable:,} / {n_total:,} "
                f"({100 * n_trainable / n_total:.2f}%)")
    return n_trainable


def _cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def info_nce_loss(image_features: torch.Tensor, text_features: torch.Tensor,
                  logit_scale: torch.Tensor) -> torch.Tensor:
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    targets = torch.arange(image_features.size(0), device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)


def train_with_coreset(cfg: dict, coreset_indices_path: str, run_name: str) -> str:
    device = get_device("cuda")
    out_dir = ensure_dir(cfg["train"]["output_dir"])

    model, preprocess, tokenizer = load_clip(
        cfg["train"]["model"], cfg["train"]["pretrained"], device
    )
    model.train()
    _unfreeze_last_n_layers(model, cfg["train"]["unfreeze_last_n_layers"])

    indices = np.load(coreset_indices_path)
    logger.info(f"Loaded coreset: {len(indices)} indices from {coreset_indices_path}")

    dataset = PABTrainDataset(
        json_path=Path(cfg["data"]["root"]) / cfg["data"]["train_json"],
        image_root=cfg["data"]["root"],
        image_transform=preprocess,
        tokenizer=tokenizer,
        subset_size=cfg["data"]["subset_size"],
        indices=indices,
        seed=cfg["seed"],
    )
    logger.info(f"Training dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    total_steps = cfg["train"]["num_epochs"] * len(loader) // cfg["train"]["grad_accum_steps"]
    warmup = cfg["train"]["warmup_steps"]
    base_lr = cfg["train"]["lr"]

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])
    use_learned_temp = cfg["train"]["use_learned_temp"]
    fixed_logit_scale = torch.tensor(1.0 / cfg["train"]["temperature"], device=device).log()

    step = 0
    for epoch in range(cfg["train"]["num_epochs"]):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg['train']['num_epochs']}")
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for it, batch in enumerate(pbar):
            images = batch["image"].to(device, non_blocking=True)
            tokens = batch["text_tokens"].to(device, non_blocking=True)

            lr = _cosine_lr(step, total_steps, warmup, base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                image_features = model.encode_image(images)
                text_features = model.encode_text(tokens)
                if use_learned_temp:
                    logit_scale = model.logit_scale.exp().clamp(max=100.0)
                else:
                    logit_scale = fixed_logit_scale.exp()
                loss = info_nce_loss(image_features, text_features, logit_scale)
                loss = loss / cfg["train"]["grad_accum_steps"]

            scaler.scale(loss).backward()
            running += loss.item() * cfg["train"]["grad_accum_steps"]

            if (it + 1) % cfg["train"]["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            if (it + 1) % cfg["train"]["log_every"] == 0:
                pbar.set_postfix(loss=f"{running / (it + 1):.4f}", lr=f"{lr:.2e}")

    ckpt_path = Path(out_dir) / f"{run_name}.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "indices_path": coreset_indices_path},
               ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")
    return str(ckpt_path)
