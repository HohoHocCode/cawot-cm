"""Paper-backed supplementary coreset baselines.

These selectors are intentionally simple adaptations that fit the current
text-image retrieval setup without changing the training objective:
  - CLIPScore filtering (Hessel et al., EMNLP 2021)
  - SemDeDup-style semantic deduplication (Abbas et al., 2023)
  - clustered k-center greedy (Sener and Savarese, ICLR 2018)
"""
from __future__ import annotations

import numpy as np

from .select import _proportional_budgets
from .utils import setup_logger

logger = setup_logger("select_published_baselines")


def _validate_budget(n_total: int, budget: int) -> int:
    budget = int(budget)
    if budget < 0 or budget > int(n_total):
        raise ValueError(f"budget must be in [0, {n_total}], got {budget}")
    return budget


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D array, got {x.shape}")
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def select_clipscore(
    zv: np.ndarray,
    zt: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Select top image-text compatible samples by CLIP cosine score."""
    zv = np.asarray(zv, dtype=np.float32)
    zt = np.asarray(zt, dtype=np.float32)
    if zv.ndim != 2 or zt.ndim != 2 or zv.shape != zt.shape:
        raise ValueError(
            f"zv and zt must have same 2D shape, got {zv.shape} and {zt.shape}"
        )
    budget = _validate_budget(len(zv), budget)
    if budget == 0:
        return np.empty((0,), dtype=np.int64)

    rng = np.random.RandomState(seed)
    scores = np.sum(zv * zt, axis=1).astype(np.float64)
    jitter = rng.uniform(0.0, 1e-12, size=len(scores))
    selected = np.argsort(-(scores + jitter))[:budget]
    out = np.sort(selected).astype(np.int64)
    logger.info(f"[clipscore] selected {len(out)} (target {budget})")
    return out


def _joint_embeddings(zv: np.ndarray, zt: np.ndarray, lambda_image: float) -> np.ndarray:
    if not (0.0 <= lambda_image <= 1.0):
        raise ValueError(f"lambda_image must be in [0, 1], got {lambda_image}")
    lambda_text = 1.0 - lambda_image
    joint = np.concatenate(
        [np.sqrt(lambda_image) * zv, np.sqrt(lambda_text) * zt],
        axis=1,
    )
    return _normalize_rows(joint)


def _dedup_local(
    pair_emb: np.ndarray,
    budget: int,
    *,
    max_similarity: float,
    keep: str,
) -> np.ndarray:
    n = len(pair_emb)
    budget = min(int(budget), n)
    if budget <= 0:
        return np.empty((0,), dtype=np.int64)
    if not (-1.0 <= max_similarity <= 1.0):
        raise ValueError(f"max_similarity must be in [-1, 1], got {max_similarity}")

    centroid = _normalize_rows(pair_emb.mean(axis=0, keepdims=True))[0]
    centroid_sim = pair_emb @ centroid
    if keep == "hard":
        order = np.argsort(centroid_sim)
    elif keep == "easy":
        order = np.argsort(-centroid_sim)
    else:
        raise ValueError(f"keep must be 'hard' or 'easy', got {keep}")

    selected: list[int] = []
    for j in order:
        if len(selected) == budget:
            break
        j = int(j)
        if not selected:
            selected.append(j)
            continue
        max_sim = float((pair_emb[selected] @ pair_emb[j]).max())
        if max_sim <= max_similarity:
            selected.append(j)

    if len(selected) < budget:
        used = set(selected)
        for j in order:
            if len(selected) == budget:
                break
            j = int(j)
            if j not in used:
                selected.append(j)
                used.add(j)
    return np.asarray(selected, dtype=np.int64)


def select_semdedup(
    zv: np.ndarray,
    zt: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    budget: int,
    seed: int = 42,
    *,
    lambda_image: float = 0.5,
    max_similarity: float = 0.95,
    keep: str = "hard",
) -> np.ndarray:
    """SemDeDup-style duplicate pruning on joint image/text embeddings."""
    del seed
    zv = np.asarray(zv, dtype=np.float32)
    zt = np.asarray(zt, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    if zv.ndim != 2 or zt.ndim != 2 or zv.shape != zt.shape:
        raise ValueError(
            f"zv and zt must have same 2D shape, got {zv.shape} and {zt.shape}"
        )
    if centroids.ndim != 2 or centroids.shape[1] != zv.shape[1]:
        raise ValueError(
            f"centroids must have shape (K, {zv.shape[1]}), got {centroids.shape}"
        )
    if assignments.shape[0] != len(zv):
        raise ValueError("assignments length must match embeddings")
    budget = _validate_budget(len(zv), budget)
    if budget == 0:
        return np.empty((0,), dtype=np.int64)

    k = int(centroids.shape[0])
    sizes = np.bincount(assignments, minlength=k)
    budgets = _proportional_budgets(sizes, budget)
    pair_emb = _joint_embeddings(zv, zt, lambda_image=lambda_image)

    selected: list[np.ndarray] = []
    for c in range(k):
        b_c = int(budgets[c])
        if b_c == 0:
            continue
        idx = np.where(assignments == c)[0]
        if len(idx) == 0:
            continue
        if b_c >= len(idx):
            selected.append(idx)
            continue
        local = _dedup_local(
            pair_emb[idx],
            b_c,
            max_similarity=max_similarity,
            keep=keep,
        )
        selected.append(idx[local])

    out = (
        np.sort(np.concatenate(selected)).astype(np.int64)
        if selected
        else np.empty((0,), np.int64)
    )
    if len(out) != budget:
        raise RuntimeError(f"SemDeDup selected {len(out)} samples, expected {budget}")
    logger.info(f"[semdedup] selected {len(out)} (target {budget})")
    return out


def _k_center_local(
    embeddings: np.ndarray,
    centroid: np.ndarray,
    budget: int,
) -> np.ndarray:
    embeddings = _normalize_rows(embeddings)
    n = len(embeddings)
    budget = min(int(budget), n)
    if budget <= 0:
        return np.empty((0,), dtype=np.int64)

    centroid = _normalize_rows(np.asarray(centroid, dtype=np.float32).reshape(1, -1))[0]
    selected = [int(np.argmax(embeddings @ centroid))]
    min_dist = 1.0 - embeddings @ embeddings[selected[0]]
    min_dist[selected[0]] = -np.inf
    while len(selected) < budget:
        j = int(np.argmax(min_dist))
        selected.append(j)
        dist = 1.0 - embeddings @ embeddings[j]
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def select_clustered_k_center(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """Clustered k-center greedy on image embeddings."""
    del seed
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
    if centroids.ndim != 2 or centroids.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"centroids must have shape (K, {embeddings.shape[1]}), got {centroids.shape}"
        )
    if assignments.shape[0] != len(embeddings):
        raise ValueError("assignments length must match embeddings")
    budget = _validate_budget(len(embeddings), budget)
    if budget == 0:
        return np.empty((0,), dtype=np.int64)

    k = int(centroids.shape[0])
    sizes = np.bincount(assignments, minlength=k)
    budgets = _proportional_budgets(sizes, budget)

    selected: list[np.ndarray] = []
    for c in range(k):
        b_c = int(budgets[c])
        if b_c == 0:
            continue
        idx = np.where(assignments == c)[0]
        if len(idx) == 0:
            continue
        if b_c >= len(idx):
            selected.append(idx)
            continue
        local = _k_center_local(embeddings[idx], centroids[c], b_c)
        selected.append(idx[local])

    out = (
        np.sort(np.concatenate(selected)).astype(np.int64)
        if selected
        else np.empty((0,), np.int64)
    )
    if len(out) != budget:
        raise RuntimeError(f"k-center selected {len(out)} samples, expected {budget}")
    logger.info(f"[k_center] selected {len(out)} (target {budget})")
    return out
