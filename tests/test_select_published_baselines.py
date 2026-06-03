import numpy as np

from src.select_published_baselines import (
    select_clipscore,
    select_clustered_k_center,
    select_semdedup,
)


def test_select_clipscore_takes_highest_image_text_alignment():
    zv = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    zt = np.array(
        [
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    selected = select_clipscore(zv, zt, budget=2, seed=0)

    assert selected.dtype == np.int64
    assert selected.tolist() == [0, 1]


def test_select_semdedup_skips_near_duplicates_then_fills_exact_budget():
    zv = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    zt = zv.copy()
    centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    assignments = np.array([0, 0, 0, 0], dtype=np.int64)

    selected = select_semdedup(
        zv,
        zt,
        centroids,
        assignments,
        budget=3,
        seed=0,
        lambda_image=0.5,
        max_similarity=0.95,
        keep="hard",
    )

    assert selected.dtype == np.int64
    assert len(selected) == 3
    assert len(set(selected.tolist())) == 3
    assert not ({0, 1} <= set(selected.tolist()))


def test_select_clustered_k_center_spreads_points_within_cluster():
    emb = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    centroids = np.array([[0.0, 1.0]], dtype=np.float32)
    assignments = np.array([0, 0, 0, 0], dtype=np.int64)

    selected = select_clustered_k_center(
        emb,
        centroids,
        assignments,
        budget=2,
        seed=0,
    )

    assert selected.dtype == np.int64
    assert selected.tolist() == [2, 3]
