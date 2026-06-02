"""FAISS k-means clustering on (image) embeddings.

For spherical k-means on CLIP embeddings: L2-normalize then use inner product.
Embeddings from embed.py are already normalized.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import setup_logger

logger = setup_logger("cluster")


def kmeans_faiss(
    embeddings: np.ndarray,
    k: int,
    niter: int = 25,
    spherical: bool = True,
    use_gpu: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (centroids [k, d], assignments [n]).

    embeddings: (n, d) float32, L2-normalized if spherical.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError("faiss not installed; pip install faiss-gpu-cu12 or faiss-cpu") from e

    n, d = embeddings.shape
    if n == 0:
        raise ValueError("kmeans_faiss needs at least one embedding")
    if k > n:
        logger.warning(f"k={k} > n={n}; reducing k to n")
        k = n
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    has_gpu = False
    if use_gpu:
        try:
            has_gpu = faiss.get_num_gpus() > 0
        except Exception:
            has_gpu = False

    logger.info(f"FAISS k-means: n={n}, d={d}, k={k}, niter={niter}, "
                f"spherical={spherical}, gpu={has_gpu}")

    kmeans = faiss.Kmeans(
        d=d,
        k=k,
        niter=niter,
        verbose=True,
        spherical=spherical,
        gpu=has_gpu,
        seed=seed,
        max_points_per_centroid=10_000_000,  # don't subsample
    )
    kmeans.train(embeddings)
    centroids = kmeans.centroids.copy()

    # Assign each point to nearest centroid
    if spherical:
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    if has_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(centroids)
    _, I = index.search(embeddings, 1)
    assignments = I.reshape(-1).astype(np.int64)
    return centroids, assignments


def hierarchical_kmeans_faiss(
    embeddings: np.ndarray,
    k_coarse: int,
    k_fine: int,
    niter: int = 25,
    spherical: bool = True,
    use_gpu: bool = True,
    seed: int = 42,
) -> dict:
    """Two-level FAISS k-means for V2.1.

    Returns a dict with:
      coarse_centroids: (K1, d)
      coarse_assignments: (n,)
      fine_centroids: (K1, K2, d); invalid slots are zeros
      fine_assignments: (n,), local fine id within each coarse cluster
      fine_valid: (K1, K2) bool mask for valid fine centroids
    """
    n, d = embeddings.shape
    if k_coarse <= 0 or k_fine <= 0:
        raise ValueError("k_coarse and k_fine must be positive")
    coarse_centroids, coarse_assignments = kmeans_faiss(
        embeddings,
        k=k_coarse,
        niter=niter,
        spherical=spherical,
        use_gpu=use_gpu,
        seed=seed,
    )
    actual_k_coarse = coarse_centroids.shape[0]
    fine_centroids = np.zeros((actual_k_coarse, k_fine, d), dtype=np.float32)
    fine_assignments = np.full(n, -1, dtype=np.int64)
    fine_valid = np.zeros((actual_k_coarse, k_fine), dtype=bool)

    for c in range(actual_k_coarse):
        idx = np.where(coarse_assignments == c)[0]
        if len(idx) == 0:
            continue
        local_k = min(k_fine, len(idx))
        if local_k == 1:
            centroid = embeddings[idx].mean(axis=0, keepdims=True).astype(np.float32)
            if spherical:
                centroid = centroid / np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-12)
            local_centroids = centroid
            local_assignments = np.zeros(len(idx), dtype=np.int64)
        else:
            local_centroids, local_assignments = kmeans_faiss(
                embeddings[idx],
                k=local_k,
                niter=niter,
                spherical=spherical,
                use_gpu=use_gpu,
                seed=seed + 1009 * c,
            )
        fine_centroids[c, :local_k] = local_centroids.astype(np.float32)
        fine_assignments[idx] = local_assignments
        fine_valid[c, :local_k] = True

    if np.any(fine_assignments < 0):
        missing = int((fine_assignments < 0).sum())
        raise RuntimeError(f"hierarchical clustering left {missing} samples unassigned")

    logger.info(
        f"hierarchical k-means: n={n}, d={d}, k_coarse={actual_k_coarse}, "
        f"k_fine={k_fine}, leaves={int(fine_valid.sum())}"
    )
    return {
        "coarse_centroids": coarse_centroids,
        "coarse_assignments": coarse_assignments,
        "fine_centroids": fine_centroids,
        "fine_assignments": fine_assignments,
        "fine_valid": fine_valid,
    }


def save_clusters(path: str | Path, centroids: np.ndarray, assignments: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=centroids, assignments=assignments)
    logger.info(f"Saved clusters to {path}")


def load_clusters(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["centroids"], data["assignments"]
