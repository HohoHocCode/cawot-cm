"""Retrieval evaluation.

Primary path (V0 sanity):
    Image-text retrieval on a held-out (image, caption) val split.
    Ground truth: caption[i] matches image[i] (1-to-1 by index).
    No person id required. This is the standard image-text retrieval setup
    used in CLIP / BLIP / ALBEF papers for cross-modal alignment.

Legacy path (for real PAB / gallery+query JSON):
    Person-id matching. Kept for compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import PABGalleryDataset, PABQueryDataset
from .embed import load_clip
from .utils import ensure_dir, get_device, setup_logger

logger = setup_logger("eval")


# -----------------------------------------------------------------------------
# Image-text retrieval on held-out val (V0 PRIMARY)
# -----------------------------------------------------------------------------


@torch.no_grad()
def _encode_image_text_pairs(model, loader, device, amp: bool):
    """Encode each (image, caption) pair. Returns (img_feats, txt_feats) both
    L2-normalized, in dataloader order."""
    img_feats, txt_feats = [], []
    for batch in tqdm(loader, desc="encode val"):
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["text_tokens"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            f_img = F.normalize(model.encode_image(images).float(), dim=-1)
            f_txt = F.normalize(model.encode_text(tokens).float(), dim=-1)
        img_feats.append(f_img.cpu().numpy())
        txt_feats.append(f_txt.cpu().numpy())
    return np.concatenate(img_feats, 0), np.concatenate(txt_feats, 0)


def _ranks_from_sims(sims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (t2i_rank, i2t_rank), each 0-indexed rank of the correct match.

    sims: (N, N) where sims[i,j] = sim(text_i, image_j). Diagonal is correct pair.
    rank = number of distractors strictly outscoring the diagonal entry.
    O(N²) without full argsort; scales to 5K+ gallery.
    """
    diag = np.diag(sims)
    t2i_rank = (sims > diag[:, None]).sum(axis=1)    # per query (row)
    i2t_rank = (sims > diag[None, :]).sum(axis=0)    # per query (column)
    return t2i_rank, i2t_rank


def _recall_from_ranks(t2i_rank, i2t_rank, mask, k_values) -> dict:
    """R@k on the subset of queries indicated by boolean `mask`.

    Gallery is always the FULL pool — we just filter which queries we average
    over. This means per-category numbers measure "given a query of category C,
    can we find its correct image in the full mixed gallery?" (matches the
    realistic retrieval setting).
    """
    n = int(mask.sum())
    if n == 0:
        return {"n_pairs": 0}
    t2i = t2i_rank[mask]
    i2t = i2t_rank[mask]
    out: dict = {"n_pairs": n}
    for k in k_values:
        out[f"t2i_R@{k}"] = float((t2i < k).mean() * 100.0)
        out[f"i2t_R@{k}"] = float((i2t < k).mean() * 100.0)
    out["mean_R@1"] = 0.5 * (out["t2i_R@1"] + out["i2t_R@1"])
    return out


def _i2t_t2i_recall(sims: np.ndarray, k_values=(1, 5, 10)) -> dict:
    """Overall image-text retrieval R@k (no category split)."""
    n = sims.shape[0]
    t2i_rank, i2t_rank = _ranks_from_sims(sims)
    out = _recall_from_ranks(t2i_rank, i2t_rank, np.ones(n, dtype=bool), k_values)
    return out


def evaluate_val_split(
    cfg: dict,
    val_dataset: Dataset,
    checkpoint_path: str | None,
    run_name: str,
) -> dict:
    """Image-text retrieval R@k on a held-out val set, with per-category split.

    If `val_dataset.categories` is set (array of length N with category strings,
    e.g. "goal"/"full"/"wentwrong"), also reports R@k filtered to each category
    (queries-only filter; gallery stays the full N images → measures realistic
    retrieval in the mixed pool).

    Returns dict: {"overall": {...}, "<cat>": {...}, ...}.
    """
    device = get_device("cuda")
    out_dir = ensure_dir(cfg["eval"]["output_dir"])

    model, _, _ = load_clip(
        cfg["train"]["model"], cfg["train"]["pretrained"], device
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        logger.info("evaluating zero-shot CLIP (no checkpoint)")
    model.eval()

    loader = DataLoader(
        val_dataset,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    img_feats, txt_feats = _encode_image_text_pairs(model, loader, device, cfg["train"]["amp"])
    sims = txt_feats @ img_feats.T
    n = sims.shape[0]
    t2i_rank, i2t_rank = _ranks_from_sims(sims)
    k_values = tuple(cfg["eval"]["k_values"])

    metrics: dict = {
        "overall": _recall_from_ranks(t2i_rank, i2t_rank, np.ones(n, dtype=bool), k_values),
    }

    categories = getattr(val_dataset, "categories", None)
    if categories is not None:
        for cat in sorted(set(categories.tolist())):
            mask = categories == cat
            metrics[cat] = _recall_from_ranks(t2i_rank, i2t_rank, mask, k_values)

    # Short log line
    summary = " | ".join(
        f"{cat}: R@1={m.get('mean_R@1', 0.0):.2f} (n={m['n_pairs']})"
        for cat, m in metrics.items()
    )
    logger.info(f"[{run_name}] {summary}")

    out_json = Path(out_dir) / f"{run_name}_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# -----------------------------------------------------------------------------
# Legacy: person-id retrieval on real gallery+queries JSON
# -----------------------------------------------------------------------------


@torch.no_grad()
def _encode_gallery(model, loader, device, amp: bool):
    feats, pids = [], []
    for batch in tqdm(loader, desc="encode gallery"):
        with torch.cuda.amp.autocast(enabled=amp):
            f = F.normalize(model.encode_image(batch["image"].to(device, non_blocking=True)).float(), dim=-1)
        feats.append(f.cpu().numpy())
        pids.append(batch["pid"].numpy())
    return np.concatenate(feats, 0), np.concatenate(pids, 0)


@torch.no_grad()
def _encode_queries(model, loader, device, amp: bool):
    feats, pids = [], []
    for batch in tqdm(loader, desc="encode queries"):
        with torch.cuda.amp.autocast(enabled=amp):
            f = F.normalize(model.encode_text(batch["text_tokens"].to(device, non_blocking=True)).float(), dim=-1)
        feats.append(f.cpu().numpy())
        pids.append(batch["pid"].numpy())
    return np.concatenate(feats, 0), np.concatenate(pids, 0)


def compute_metrics(sims: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray,
                    k_values=(1, 5, 10)) -> dict:
    Q, G = sims.shape
    sorted_idx = np.argsort(-sims, axis=1)
    sorted_pids = g_pids[sorted_idx]
    matches = (sorted_pids == q_pids[:, None]).astype(np.float32)
    out: dict = {}
    for k in k_values:
        out[f"R@{k}"] = float((matches[:, :k].sum(1) > 0).mean() * 100.0)
    ap_per_q = []
    for q in range(Q):
        m = matches[q]
        if m.sum() == 0:
            continue
        ranks = np.arange(1, G + 1)
        ap = ((np.cumsum(m) / ranks) * m).sum() / m.sum()
        ap_per_q.append(ap)
    out["mAP"] = float(np.mean(ap_per_q) * 100.0) if ap_per_q else 0.0
    out["num_queries"] = int(Q)
    out["num_gallery"] = int(G)
    return out


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
        logger.info(f"loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        logger.info("evaluating zero-shot CLIP")
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
    sims = q_feat @ g_feat.T
    metrics = compute_metrics(sims, q_pid, g_pid, k_values=tuple(cfg["eval"]["k_values"]))
    logger.info(f"[{run_name}] {metrics}")
    with open(Path(out_dir) / f"{run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
