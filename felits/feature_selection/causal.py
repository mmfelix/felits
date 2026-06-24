"""Causal feature selection via Granger causality and PCMCI++ (stub).

The module exposes:

- :func:`granger_causality_test` — bivariate Granger test wrapped around
  :class:`statsmodels.tsa.stattools.grangercausalitytests`.
- :func:`granger_feature_selection` — a multi-feature helper that
  applies the bivariate test to each candidate feature and returns the
  list of features with statistically significant causality.
- :func:`pcmci_selection` — a thin stub around the PCMCI++ algorithm
  from the ``tigramite`` package. It is included for reference and to
  document the API; it imports ``tigramite`` lazily so the rest of the
  module is usable without it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning as StatsmodelsValueWarning
from statsmodels.tsa.stattools import grangercausalitytests

__all__ = [
    "GrangerResult",
    "granger_causality_test",
    "granger_feature_selection",
    "pcmci_selection",
]


@dataclass
class GrangerResult:
    """Result of a single bivariate Granger causality test."""

    feature: str
    target: str
    lag: int
    p_value: float
    significant: bool


def granger_causality_test(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    max_lag: int,
    significance: float = 0.05,
) -> list[GrangerResult]:
    """Run Granger causality tests of ``x`` causing ``y`` for each lag 1..max_lag.

    Parameters
    ----------
    x, y:
        The candidate "cause" and "effect" series, respectively. They must
        be 1-D, the same length, and contain no NaNs.
    max_lag:
        Maximum number of lags to test.
    significance:
        p-value threshold for declaring a feature as significant.

    Returns
    -------
    list[GrangerResult]
        One entry per lag. The ``significant`` flag is ``True`` when
        ``p_value < significance``.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError("`x` and `y` must have the same length.")
    if x_arr.size < max_lag * 4:
        raise ValueError("Series too short for the requested `max_lag`.")
    # statsmodels expects the two columns as a 2-D array with [y, x] order.
    data = np.column_stack([y_arr, x_arr])
    # statsmodels' Granger test fits an OLS regression of y on its own lags
    # plus x's lags. When ``x`` is itself a lag/rolling version of ``y`` (the
    # common case for our engineered candidate features) the design matrix
    # is near-collinear and statsmodels emits:
    #   ``ValueWarning: covariance of constraints does not have full rank``
    # The F-test p-value is still computed and valid; the warning is purely
    # informational. Suppress it locally to keep the prepare_data.py log
    # readable. Other statsmodels warnings (e.g. singular matrices) are kept.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"covariance of constraints does not have full rank.*",
            category=StatsmodelsValueWarning,
        )
        # The ``verbose`` keyword is deprecated in recent statsmodels; the
        # F-test results are returned regardless of its value, so silence
        # the deprecation noise while still passing ``verbose=False``.
        warnings.filterwarnings(
            "ignore",
            message=r".*verbose.*deprecated.*",
            category=FutureWarning,
        )
        raw = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    out: list[GrangerResult] = []
    for lag, res in raw.items():
        # res[0] is the ssr-based F test, res[1] the ssr chi2 test, etc.
        p_value = float(res[0]["ssr_ftest"][1])
        out.append(
            GrangerResult(
                feature="x",
                target="y",
                lag=int(lag),
                p_value=p_value,
                significant=p_value < significance,
            )
        )
    return out


def granger_feature_selection(
    df: pd.DataFrame,
    target: str,
    max_lag: int,
    significance: float = 0.05,
    candidates: list[str] | None = None,
) -> list[str]:
    """Return the features in ``df`` that Granger-cause ``target``.

    Parameters
    ----------
    df:
        DataFrame sorted by time.
    target:
        Name of the target column.
    max_lag:
        Maximum lag to test.
    significance:
        p-value threshold.
    candidates:
        Subset of columns to consider. ``None`` uses all numeric columns
        other than ``target``.
    """
    if target not in df.columns:
        raise KeyError(f"target={target!r} not in DataFrame columns.")
    if candidates is None:
        candidates = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    y = df[target].to_numpy(dtype=float)
    selected: list[str] = []
    for col in candidates:
        x = df[col].to_numpy(dtype=float)
        try:
            res = granger_causality_test(x, y, max_lag=max_lag, significance=significance)
        except ValueError:
            continue
        if any(r.significant for r in res):
            selected.append(col)
    return selected


def pcmci_selection(
    df: pd.DataFrame,
    target: str,
    max_lag: int = 1,
    significance: float = 0.05,
) -> list[str]:
    """PCMCI++ feature selection (Runge 2020).

    This is a thin wrapper around ``tigramite``'s PCMCI implementation.
    The package is an optional dependency; the function raises an
    informative ``ImportError`` when it is not installed.
    """
    try:
        from tigramite.independence_test.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise ImportError(
            "pcmci_selection requires tigramite. Install with: pip install tigramite"
        ) from exc

    data = df.to_numpy(dtype=float)
    var_names = list(df.columns)
    pcmci = PCMCI(dataframe=ParCorr(), verbosity=0)
    res = pcmci.run_pcmciplus(data=data, tau_min=0, tau_max=max_lag, pc_alpha=significance)
    # parents of `target`:
    target_idx = var_names.index(target)
    parents: dict[int, dict[int, list[int]]] = res["parents"]
    selected = {
        var_names[p]
        for j, pmap in parents.items()
        if j == target_idx
        for parents_at_lag, p_list in pmap.items()
        for p in p_list
    }
    selected.discard(target)
    return sorted(selected)
