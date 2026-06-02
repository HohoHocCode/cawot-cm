import numpy as np

from src.select import wasserstein_aware_budgets
from src.select_v2_1 import (
    _facility_location_gain,
    exact_w2_1d_squared,
    prototype_conditioned_greedy,
    select_v2_1,
    sliced_wasserstein_distance,
)


def test_exact_w2_1d_squared_handles_unequal_empirical_sizes():
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 1.0, 1.0])

    assert exact_w2_1d_squared(x, y) == 1.0


def test_sliced_wasserstein_distance_detects_shift_with_fixed_projections():
    x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    y = np.array([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    projections = np.array([[1.0, 0.0]], dtype=np.float32)

    assert sliced_wasserstein_distance(x, y, projections=projections) == 2.0


def test_wasserstein_aware_budgets_redistributes_after_capacity_caps():
    sizes = np.array([1, 100, 100])
    gaps = np.array([1000.0, 1.0, 1.0])

    budgets = wasserstein_aware_budgets(sizes, gaps, total_budget=50, alpha=0.5)

    assert budgets.sum() == 50
    assert budgets[0] == 1
    assert np.all(budgets <= sizes)


def test_facility_location_gain_uses_sum_normalized_by_cluster_size():
    sim = np.array(
        [
            [1.0, 0.2, 0.6],
            [0.4, 1.0, 0.5],
            [0.1, 0.3, 1.0],
        ],
        dtype=np.float64,
    )
    coverage = np.array([0.3, 0.2, 0.8], dtype=np.float64)

    gains = _facility_location_gain(sim, coverage)

    expected = np.maximum(sim - coverage[:, None], 0.0).sum(axis=0) / sim.shape[0]
    assert np.allclose(gains, expected)
    assert np.all((0.0 <= gains) & (gains <= 1.0))


def test_prototype_conditioned_greedy_alpha_one_matches_top_prototypes():
    sim = np.eye(4, dtype=np.float64)
    proto = np.array([0.2, 0.9, 0.5, 0.7], dtype=np.float64)

    selected = prototype_conditioned_greedy(sim, proto, k=2, alpha=1.0)

    assert selected.tolist() == [1, 3]


def test_select_v2_1_uses_text_centroid_when_lambda_image_is_zero():
    zv = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    zt = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    q = zt.copy()
    coarse_assignments = np.array([0, 0, 0])
    fine_assignments = np.array([0, 0, 0])
    coarse_centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    fine_centroids = np.array([[[1.0, 0.0]]], dtype=np.float32)
    fine_valid = np.array([[True]])

    selected = select_v2_1(
        zv=zv,
        zt=zt,
        coarse_centroids=coarse_centroids,
        coarse_assignments=coarse_assignments,
        fine_centroids=fine_centroids,
        fine_assignments=fine_assignments,
        fine_valid=fine_valid,
        q_proxy_emb=q,
        budget=1,
        alpha=1.0,
        lambda_image=0.0,
        num_projections=1,
        seed=3,
    )

    assert selected.tolist() == [2]


def test_select_v2_1_returns_exact_budget_and_uses_hierarchy():
    zv = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
            [-1.0, 0.0],
            [-0.95, -0.05],
        ],
        dtype=np.float32,
    )
    zt = zv.copy()
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    coarse_assignments = np.array([0, 0, 0, 0, 1, 1])
    fine_assignments = np.array([0, 0, 1, 1, 0, 0])
    coarse_centroids = np.array([[0.7, 0.7], [-1.0, 0.0]], dtype=np.float32)
    fine_centroids = np.zeros((2, 2, 2), dtype=np.float32)
    fine_centroids[0, 0] = [1.0, 0.0]
    fine_centroids[0, 1] = [0.0, 1.0]
    fine_centroids[1, 0] = [-1.0, 0.0]
    fine_valid = np.array([[True, True], [True, False]])

    selected = select_v2_1(
        zv=zv,
        zt=zt,
        coarse_centroids=coarse_centroids,
        coarse_assignments=coarse_assignments,
        fine_centroids=fine_centroids,
        fine_assignments=fine_assignments,
        fine_valid=fine_valid,
        q_proxy_emb=q,
        budget=4,
        alpha=0.75,
        lambda_image=0.7,
        num_projections=8,
        seed=7,
    )

    assert selected.shape == (4,)
    assert selected.dtype == np.int64
    assert len(set(selected.tolist())) == 4
    assert np.all((0 <= selected) & (selected < len(zv)))
