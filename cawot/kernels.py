"""
cawot/kernels.py
=================
Kernel definitions and scalable MMD estimation for finite-proxy query-aware
coreset selection.

Design decisions (frozen, see plan §3 and §12):
  * CLIP embeddings are L2-normalized and live on the unit sphere. We use the
    restriction of an ambient Gaussian RBF kernel to the sphere. Because for
    unit vectors ||x - y||^2 = 2 - 2 cos(x, y), this RBF is a monotone function
    of cosine similarity -- it is aligned with CLIP geometry while admitting the
    standard Rahimi-Recht random-feature approximation.
  * MMD is estimated through Random Fourier Features (RFF), NOT a dense kernel
    matrix. The RFF estimate ||mean(psi(X)) - mean(psi(Y))||^2 is a mean
    embedding in random-feature space approximating the RKHS -- it is NOT the
    Euclidean distance between CLIP centroids (which would collapse MMD to a
    first-moment statistic).

All randomness is seeded through an explicit numpy Generator for reproducibility.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Bandwidth                                                                   #
# --------------------------------------------------------------------------- #
def median_heuristic_bandwidth(
    X: np.ndarray,
    max_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Median-of-pairwise-distances bandwidth (sigma) for a Gaussian RBF kernel.

    Uses at most ``max_samples`` points to keep the pairwise computation cheap.
    Returns sigma such that k(x, y) = exp(-||x - y||^2 / (2 sigma^2)).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = X.shape[0]
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    # pairwise squared Euclidean distances
    sq = np.sum(Xs**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Xs @ Xs.T)
    iu = np.triu_indices(Xs.shape[0], k=1)
    med = np.median(d2[iu])
    med = float(max(med, 1e-12))
    # sigma^2 = med / 2  (a common convention); return sigma
    return float(np.sqrt(med / 2.0))


# --------------------------------------------------------------------------- #
#  Random Fourier Features                                                      #
# --------------------------------------------------------------------------- #
class RFF:
    """Random Fourier Features for the Gaussian RBF kernel.

    k(x, y) = exp(-||x - y||^2 / (2 sigma^2))  is approximated by
    psi(x) = sqrt(2/D) * cos(W x + b),  W ~ N(0, sigma^-2 I),  b ~ U[0, 2pi].

    The feature map satisfies  E[psi(x)^T psi(y)] = k(x, y).
    """

    def __init__(self, dim: int, n_features: int, sigma: float,
                 rng: np.random.Generator | None = None):
        rng = np.random.default_rng() if rng is None else rng
        self.dim = dim
        self.n_features = n_features
        self.sigma = float(sigma)
        # W has rows ~ N(0, (1/sigma^2) I)
        self.W = rng.normal(loc=0.0, scale=1.0 / self.sigma,
                            size=(n_features, dim)).astype(np.float32)
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=n_features).astype(np.float32)
        self._scale = np.float32(np.sqrt(2.0 / n_features))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map (n, dim) -> (n, n_features)."""
        X = np.asarray(X, dtype=np.float32)
        proj = X @ self.W.T + self.b  # (n, D)
        return self._scale * np.cos(proj)

    def mean_embedding(self, X: np.ndarray) -> np.ndarray:
        """Empirical mean feature vector  (1/n) sum_i psi(x_i)  -> (n_features,)."""
        return self.transform(X).mean(axis=0)


# --------------------------------------------------------------------------- #
#  MMD^2 via RFF                                                                #
# --------------------------------------------------------------------------- #
def mmd2_rff(mu_p: np.ndarray, mu_q: np.ndarray) -> float:
    """Squared MMD between two distributions given their RFF mean embeddings.

    MMD^2(P, Q) ~= || mu_P - mu_Q ||_2^2  in random-feature space.

    NOTE: this is a *biased* plug-in estimator (consistent as n, m, D grow). It
    is the quantity Theorem 1 concentrates around (raw score d_{c,r}). We keep it
    simple and deterministic given the RFF map; the small positive bias is
    discussed in the plan and absorbed by the per-proxy normalization step.
    """
    diff = mu_p - mu_q
    return float(diff @ diff)


def mmd2_rff_from_points(psi_X: np.ndarray, psi_Y: np.ndarray) -> float:
    """Convenience: squared MMD from already-transformed feature matrices."""
    return mmd2_rff(psi_X.mean(axis=0), psi_Y.mean(axis=0))


# --------------------------------------------------------------------------- #
#  Additive pair-kernel feature map (for clustering & within-cluster herding)  #
# --------------------------------------------------------------------------- #
class PairFeatureMap:
    """Explicit feature map for the additive pair kernel.

        k_pair((v,t),(v',t')) = lam * k_v(v,v') + (1-lam) * k_t(t,t')

    Implemented as the concatenation  [sqrt(lam) * psi_v(v), sqrt(1-lam) * psi_t(t)]
    so that the inner product reproduces the additive kernel.

    IMPORTANT (see plan, step 2): this concatenated vector is ONLY an
    implementation of a kernel feature map for clustering / within-cluster
    selection. It is NOT a fused pair representation used for query-relevance
    scoring (relevance is text-side only).
    """

    def __init__(self, rff_v: RFF, rff_t: RFF, lam: float = 0.7):
        assert 0.0 <= lam <= 1.0
        self.rff_v = rff_v
        self.rff_t = rff_t
        self.lam = float(lam)

    def transform(self, Zv: np.ndarray, Zt: np.ndarray) -> np.ndarray:
        pv = self.rff_v.transform(Zv) * np.float32(np.sqrt(self.lam))
        pt = self.rff_t.transform(Zt) * np.float32(np.sqrt(1.0 - self.lam))
        return np.concatenate([pv, pt], axis=1)
