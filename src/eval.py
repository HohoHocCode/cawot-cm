"""Retrieval evaluation: text-to-image on PAB test split.

Metrics: R@1, R@5, R@10, mAP (mean Average Precision).

PAB: 1978 queries vs 36773 gallery. Match by person id (pid).
A retrieved gallery image counts as correct if its pid equals the query's pid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PABGalleryDataset, PABQueryDataset
from .embed import load_clip
from .utils import ensure_dir, get_device, setup_logger

logger = setup_logger("eval")


@torch.no_grad()
def _encode_gallery(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    feats, pids = [], []
    for batch in tqdm(loader, desc="encode gallery"):
        with torch.cuda.amp.autocast(enabled=amp):
            f = model.encode_image(batch["image"].to(device, non_blocking=True))
            f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu().numpy())
        pids.append(batch["pid"].numpy())
    return np.concatenate(feats, axis=0), np.concatenate(pids, axis=0)


@torch.no_grad()
def _encode_queries(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    feats, pids = [], []
    for batch in tqdm(loader, desc="encode queries"):
        with torch.cuda.amp.autocast(enabled=amp):
            f = model.encode_text(batch["text_tokens"].to(device, non_blocking=True))
            f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu().numpy())
        pids.append(batch["pid"].numpy())
    return np.concatenate(feats, axis=0), np.concatenate(pids, axis=0)


def compute_metrics(sims: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray,
                    k_values=(1, 5, 10)) -> dict:
    """sims: (Q, G) cosine similarities. Higher = more similar."""
    Q, G = sims.shape
    # Sort gallery indices by descending similarity per query
    sorted_idx = np.argsort(-sims, axis=1)
    sorted_pids = g_pids[sorted_idx]  # (Q, G)
    matches = (sorted_pids == q_pids[:, None]).astype(np.float32)  # (Q, G)

    metrics = {}
    for k in k_values:
        recall_at_k = (matches[:, :k].sum(axis=1) > 0).astype(np.float32).mean()
        metrics[f"R@{k}"] = float(recall_at_k * 100.0)

    # mAP
    ap_per_q = []
    for q in range(Q):
        m = matches[q]
        if m.sum() == 0:
            continue
        ranks = np.arange(1, G + 1)
        cum_hits = np.cumsum(m)
        precision_at_rank = cum_hits / ranks
        ap = (precision_at_rank * m).sum() / m.sum()
        ap_per_q.append(ap)
    metrics["mAP"] = float(np.mean(ap_per_q) * 100.0) if ap_per_q else 0.0
    metrics["num_queries"] = int(Q)
    metrics["num_gallery"] = int(G)
    return metrics


def evaluate(cfg: dict, checkpoint_path: str, run_name: str) -> dict:
    device = get_device("cuda")
    out_dir = ensure_dir(cfg["eval"]["output_dir"])

    model, preprocess, tokenizer = load_clip(
        cfg["train"]["model"], cfg["train"]["pretrained"], device
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        logger.info("No checkpoint provided — evaluating zero-shot CLIP")
    model.eval()

    root = Path(cfg["data"]["root"])
    gallery_ds = PABGalleryDataset(root / cfg["data"]["test_gallery_json"], root, preprocess)
    query_ds = PABQueryDataset(root / cfg["data"]["test_query_json"], tokenizer)

    g_loader = DataLoader(gallery_ds, batch_size=cfg["eval"]["batch_size"],
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    q_loader = DataLoader(query_ds, batch_size=cfg["eval"]["batch_size"],
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)

    amp = cfg["train"]["amp"]
    g_feat, g_pid = _encode_gallery(model, g_loader, device, amp)
    q_feat, q_pid = _encode_queries(model, q_loader, device, amp)
    logger.info(f"gallery: {g_feat.shape}  queries: {q_feat.shape}")

    sims = q_feat @ g_feat.T   # (Q, G), both L2-normalized → cosine
    metrics = compute_metrics(sims, q_pid, g_pid, k_values=tuple(cfg["eval"]["k_values"]))
    logger.info(f"[{run_name}] metrics: {metrics}")

    out_json = Path(out_dir) / f"{run_name}_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
