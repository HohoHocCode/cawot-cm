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


def save_clusters(path: str | Path, centroids: np.ndarray, assignments: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=centroids, assignments=assignments)
    logger.info(f"Saved clusters to {path}")


def load_clusters(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["centroids"], data["assignments"]
