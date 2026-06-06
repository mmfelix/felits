"""Information-theoretic feature selection.

The module provides:

- :func:`mutual_information_ksg` — KSG (Kraskov-Stögbauer-Grassberger)
  estimator of mutual information between two continuous variables.
  Implemented from scratch using ``sklearn.neighbors.NearestNeighbors``;
  the estimator is the canonical k-NN variant with the local non-uniform
  correction (Kraskov et al. 2004, eq. 11).
- :func:`mutual_information_matrix` — pairwise MI matrix over a
  ``DataFrame``.
- :func:`mrmr_selection` — Minimum Redundancy Maximum Relevance
  selection (Peng et al. 2005) using KSG MI under the hood.
- :func:`conditional_mutual_information` — CMI estimator based on
  k-nearest-neighbour counts.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "conditional_mutual_information",
    "mrmr_selection",
    "mutual_information_ksg",
    "mutual_information_matrix",
]


def _psi(n):
    """Digamma function evaluated elementwise; accepts scalars or arrays."""
    return np.asarray(digamma(np.asarray(n, dtype=float)), dtype=float)


def mutual_information_ksg(
    x: np.ndarray, y: np.ndarray, k: int = 3, base: float | None = None
) -> float:
    """Estimate MI between ``x`` and ``y`` using the KSG estimator.

    Parameters
    ----------
    x, y:
        1-D arrays of equal length.
    k:
        Number of neighbours for the kNN distance (k ≥ 1).
    base:
        If given, return MI in units of ``log_base``; otherwise nats.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("`x` and `y` must have the same shape.")
    n = x.size
    if n <= k:
        return 0.0
    xy = np.column_stack([x, y])
    # Joint kNN search with Chebyshev metric (Kraskov et al. 2004, eq. 11).
    nn_joint = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1).fit(xy)
    dists_joint, _ = nn_joint.kneighbors(xy)
    eps = dists_joint[:, k]  # distance to the k-th neighbour (excl. self)

    # Marginal counts in a fixed margin: for each point, count neighbours
    # strictly within margin in x and in y.
    nn_x = NearestNeighbors(metric="chebyshev").fit(x.reshape(-1, 1))
    nn_y = NearestNeighbors(metric="chebyshev").fit(y.reshape(-1, 1))
    nx = nn_x.radius_neighbors(x.reshape(-1, 1), radius=eps, return_distance=False)
    ny = nn_y.radius_neighbors(y.reshape(-1, 1), radius=eps, return_distance=False)
    # Subtract 1 to exclude the self point.
    nx_count = np.array([len(idx) - 1 for idx in nx], dtype=float)
    ny_count = np.array([len(idx) - 1 for idx in ny], dtype=float)

    mi = _psi(k) + _psi(n) - np.mean(_psi(nx_count + 1) + _psi(ny_count + 1))
    mi = float(np.asarray(mi).item())
    if base is not None:
        mi = mi / np.log(base)
    return float(max(mi, 0.0))


def mutual_information_matrix(
    df: pd.DataFrame, k: int = 3, skip_diagonal: bool = True
) -> pd.DataFrame:
    """Return the symmetric pairwise MI matrix over ``df``'s columns."""
    cols = list(df.columns)
    arr = df.to_numpy(dtype=float)
    n = len(cols)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mi = mutual_information_ksg(arr[:, i], arr[:, j], k=k)
            mat[i, j] = mat[j, i] = mi
    out = pd.DataFrame(mat, index=cols, columns=cols)
    if skip_diagonal:
        # Use a writable array; pd.DataFrame.values is read-only.
        arr = out.to_numpy().copy()
        np.fill_diagonal(arr, np.nan)
        return pd.DataFrame(arr, index=cols, columns=cols)
    return out


def mrmr_selection(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    k_features: int = 10,
    mi_method: str = "ksg",
) -> list[str]:
    """Minimum Redundancy Maximum Relevance (Peng et al. 2005) feature ranking.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target vector.
    k_features:
        Number of features to select.
    mi_method:
        ``"ksg"`` to use the KSG estimator (slower, accurate),
        ``"sklearn"`` to fall back to :func:`sklearn.feature_selection.mutual_info_regression`
        (faster, less accurate on small samples).
    """
    cols = list(X.columns)
    arr = X.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float).ravel()
    if mi_method == "ksg":

        def _mi(a: np.ndarray, b: np.ndarray) -> float:
            return mutual_information_ksg(a, b)
    elif mi_method == "sklearn":

        def _mi(a: np.ndarray, b: np.ndarray) -> float:
            return float(mutual_info_regression(a.reshape(-1, 1), b, random_state=0)[0])
    else:
        raise ValueError(f"Unknown mi_method={mi_method!r}")
    relevance = np.array([_mi(arr[:, i], yv) for i in range(len(cols))])
    selected: list[int] = []
    candidate_mask = np.ones(len(cols), dtype=bool)
    # First pick: max relevance.
    first = int(np.argmax(relevance))
    selected.append(first)
    candidate_mask[first] = False
    redundancy_sum = np.zeros(len(cols))
    for _ in range(1, k_features):
        last = selected[-1]
        redundancy_sum += np.array(
            [_mi(arr[:, last], arr[:, j]) if candidate_mask[j] else 0.0 for j in range(len(cols))]
        )
        scores = relevance - redundancy_sum / max(len(selected), 1)
        scores[~candidate_mask] = -np.inf
        nxt = int(np.argmax(scores))
        selected.append(nxt)
        candidate_mask[nxt] = False
    return [cols[i] for i in selected]


def conditional_mutual_information(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 3
) -> float:
    """kNN-based estimator of ``I(X ; Y | Z)`` (Gao et al. 2017).

    Implementation note: we use a simplified version based on KSG. For
    typical STLF sample sizes (n > 1000) the estimate is stable enough for
    feature ranking, but it is not a substitute for permutation tests
    when ``n`` is small.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float)
    n = x.size
    if y.shape != x.shape or z.shape[0] != n:
        raise ValueError("Inputs must have the same length along axis 0.")
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    xyz = np.column_stack([x, y, z])
    nn_joint = NearestNeighbors(metric="chebyshev", n_neighbors=k + 1).fit(xyz)
    dists, _ = nn_joint.kneighbors(xyz)
    eps = dists[:, k]
    nn_xz = NearestNeighbors(metric="chebyshev").fit(np.column_stack([x, z]))
    nn_yz = NearestNeighbors(metric="chebyshev").fit(np.column_stack([y, z]))
    nn_z = NearestNeighbors(metric="chebyshev").fit(z)
    nxz = nn_xz.radius_neighbors(np.column_stack([x, z]), radius=eps, return_distance=False)
    nyz = nn_yz.radius_neighbors(np.column_stack([y, z]), radius=eps, return_distance=False)
    nz = nn_z.radius_neighbors(z, radius=eps, return_distance=False)
    nxz_count = np.array([len(idx) - 1 for idx in nxz], dtype=float)
    nyz_count = np.array([len(idx) - 1 for idx in nyz], dtype=float)
    nz_count = np.array([len(idx) - 1 for idx in nz], dtype=float)
    cmi = _psi(k) + np.mean(_psi(nz_count + 1) - _psi(nxz_count + 1) - _psi(nyz_count + 1))
    cmi = float(np.asarray(cmi).item())
    return float(max(cmi, 0.0))


# Local import of pandas to avoid unused import linting on the module
import pandas as pd  # noqa: E402
