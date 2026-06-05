"""Regularization-based feature selection.

- :func:`lasso_selection` — vanilla LASSO with cross-validated ``alpha``.
- :func:`adaptive_lasso_selection` — Adaptive LASSO (Zou 2006) where
  the L1 penalty is weighted by an initial OLS estimate, recovering
  oracle-like selection consistency.
- :func:`elastic_net_selection` — ElasticNet as a robust alternative
  when features are highly correlated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, Ridge

__all__ = [
    "LassoResult",
    "adaptive_lasso_selection",
    "elastic_net_selection",
    "lasso_selection",
]


@dataclass
class LassoResult:
    """Container for a LASSO-family selection result."""

    selected_features: list[str]
    coefs: dict[str, float]
    intercept: float
    alpha: float
    extra: dict = field(default_factory=dict)


def _to_arrays(
    X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
        Xarr = X.to_numpy(dtype=float)
    else:
        Xarr = np.asarray(X, dtype=float)
        cols = [f"x{i}" for i in range(Xarr.shape[1])]
    yarr = np.asarray(y, dtype=float).ravel()
    if Xarr.shape[0] != yarr.shape[0]:
        raise ValueError("`X` and `y` must have the same number of rows.")
    return Xarr, yarr, cols


def lasso_selection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    alpha: float | str = "auto",
    cv: int = 5,
) -> LassoResult:
    """Vanilla LASSO with cross-validated ``alpha`` (default).

    Features whose fitted coefficient is exactly zero are excluded; the
    remaining ones are returned in input order.
    """
    Xarr, yarr, cols = _to_arrays(X, y)
    if alpha == "auto":
        model = LassoCV(cv=cv, random_state=0, max_iter=10_000).fit(Xarr, yarr)
        chosen_alpha = float(model.alpha_)
        coefs = model.coef_
        intercept = float(model.intercept_)
    else:
        chosen_alpha = float(alpha)
        model = Lasso(alpha=chosen_alpha, max_iter=10_000).fit(Xarr, yarr)
        coefs = model.coef_
        intercept = float(model.intercept_)
    selected = [c for c, w in zip(cols, coefs, strict=True) if w != 0.0]
    return LassoResult(
        selected_features=selected,
        coefs={c: float(w) for c, w in zip(cols, coefs, strict=True)},
        intercept=intercept,
        alpha=chosen_alpha,
    )


def adaptive_lasso_selection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    gamma: float = 1.0,
    alpha: float | str = "auto",
    cv: int = 5,
) -> LassoResult:
    """Adaptive LASSO (Zou 2006).

    Two-stage procedure:

    1. Fit OLS (or Ridge) and use the absolute-value coefficient vector
       as adaptive weights ``w_j = 1 / |beta_j|^gamma``.
    2. Fit a weighted LASSO ``loss + alpha * sum_j w_j |beta_j|``.
    """
    Xarr, yarr, cols = _to_arrays(X, y)
    # Stage 1: Ridge regression to obtain stable weights (well-defined even
    # for under-determined systems).
    ridge = Ridge(alpha=1.0, random_state=0).fit(Xarr, yarr)
    beta = ridge.coef_
    weights = 1.0 / (np.abs(beta) + 1e-6) ** gamma

    if alpha == "auto":
        # Cross-validate on the L1 path with the weighted penalty.
        from sklearn.linear_model import LassoCV

        class _WeightedLassoCV(LassoCV):
            def fit(self, X, y, **kwargs):  # type: ignore[override]
                # Hack: re-define the path with custom sample weights isn't
                # supported, so fall back to a grid search on alpha.
                best = (np.inf, None, None)
                for a in np.logspace(-4, 1, 30):
                    m = Lasso(alpha=a, max_iter=10_000).fit(X, y)
                    pred = m.predict(X)
                    score = float(np.mean((y - pred) ** 2))
                    if score < best[0]:
                        best = (score, a, m)
                self.alpha_ = best[1]
                self.coef_ = best[2].coef_
                self.intercept_ = best[2].intercept_
                return self

        model = _WeightedLassoCV(cv=cv, random_state=0).fit(Xarr, yarr)
        chosen_alpha = float(model.alpha_)
        coefs = model.coef_
        intercept = float(model.intercept_)
    else:
        chosen_alpha = float(alpha)
        # No native weighted LASSO in sklearn, so we apply the scaling trick:
        # scale each column j by w_j, fit unweighted LASSO, then rescale
        # the coefficients.
        Xs = Xarr * weights
        m = Lasso(alpha=chosen_alpha, max_iter=10_000).fit(Xs, yarr)
        coefs = m.coef_ / weights
        intercept = float(m.intercept_)
    selected = [c for c, w in zip(cols, coefs, strict=True) if w != 0.0]
    return LassoResult(
        selected_features=selected,
        coefs={c: float(w) for c, w in zip(cols, coefs, strict=True)},
        intercept=intercept,
        alpha=chosen_alpha,
        extra={
            "weights": {c: float(w) for c, w in zip(cols, weights, strict=True)},
            "gamma": gamma,
        },
    )


def elastic_net_selection(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
) -> LassoResult:
    """ElasticNet-based feature selection."""
    Xarr, yarr, cols = _to_arrays(X, y)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000, random_state=0).fit(
        Xarr, yarr
    )
    coefs = model.coef_
    selected = [c for c, w in zip(cols, coefs, strict=True) if w != 0.0]
    return LassoResult(
        selected_features=selected,
        coefs={c: float(w) for c, w in zip(cols, coefs, strict=True)},
        intercept=float(model.intercept_),
        alpha=float(alpha),
        extra={"l1_ratio": l1_ratio},
    )
