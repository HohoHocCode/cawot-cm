"""
cawot -- Finite-Proxy Query-Aware Kernel Coreset Selection
==========================================================

Compact, scalable coreset selection for cross-modal anomaly retrieval where the
deployment query distribution is known only approximately, through a finite
family of semantic proxy queries.

Backbone: mean finite-proxy text-side MMD relevance (abar_c).
Optional bonus: proxy-disagreement (u_c), controlled by eta.
Within-cluster: approximate kernel herding on a pair-kernel feature map.

See the accompanying plan document for design rationale and the honesty notes
(Theorem 1 is a statistical guarantee, not the novelty; u_c is a hypothesis to
be tested by ablation; complexity claims are for the actual algorithm).
"""
from . import kernels, clustering, scoring, selection, baselines, diagnostics, data

__all__ = [
    "kernels", "clustering", "scoring", "selection",
    "baselines", "diagnostics", "data",
]
__version__ = "0.1.0"
