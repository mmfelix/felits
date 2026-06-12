"""Automated feature-extraction backends (FATS, tsfresh)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as _stats

__all__ = [
    "tsfresh_extract",
    "fats_extract",
    "extract_all_features",
]


def tsfresh_extract(
    df: pd.DataFrame,
    column_id: str,
    column_sort: str,
    target: Optional[np.ndarray] = None,
    default_fc_parameters: Optional[dict | object] = None,
    n_jobs: int = 0,
    disable_progressbar: bool = True,
) -> pd.DataFrame:
    """Run tsfresh feature extraction (requires pandas DataFrame internally)."""
    try:
        from tsfresh import extract_features as _tsf_extract
        from tsfresh import select_features as _tsf_select
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError as exc:
        raise ImportError("tsfresh_extract requires tsfresh.") from exc

    if default_fc_parameters is None:
        default_fc_parameters = MinimalFCParameters()
    features = _tsf_extract(
        df,
        column_id=column_id,
        column_sort=column_sort,
        default_fc_parameters=default_fc_parameters,
        n_jobs=n_jobs,
        disable_progressbar=disable_progressbar,
    )
    if target is not None:
        unique_ids = df[column_id].unique()
        target_aligned = (
            target.loc[unique_ids] if hasattr(target, "loc") else target[: len(unique_ids)]
        )
        features = _tsf_select(features, target_aligned)
    return features


def fats_extract(series, n_periods: int = 1) -> dict[str, float]:
    """Compute FATS-style scalar features on a 1-D series."""
    arr = np.asarray(series, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return {
            k: float("nan")
            for k in [
                "amplitude",
                "beyond_1_std",
                "beyond_2_std",
                "car_sigma",
                "car_mean",
                "car_std",
                "skew",
                "kurtosis",
                "median_abs_dev",
            ]
        }
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if sigma == 0:
        sigma = float(np.std(arr))
    return {
        "amplitude": float(arr.max() - arr.min()),
        "beyond_1_std": float(np.mean(np.abs(arr - arr.mean()) > 1.0 * arr.std())),
        "beyond_2_std": float(np.mean(np.abs(arr - arr.mean()) > 2.0 * arr.std())),
        "car_sigma": float(arr.std() / max(abs(arr.mean()), 1e-12)),
        "car_mean": float(arr.mean()),
        "car_std": float(arr.std()),
        "skew": float(_stats.skew(arr)),
        "kurtosis": float(_stats.kurtosis(arr)),
        "median_abs_dev": float(mad),
    }


def extract_all_features(
    df: pd.DataFrame,
    target: str,
    *,
    add_cyclic: bool = True,
    cyclic_period: int = 24,
    add_rolling: bool = True,
    rolling_windows: tuple[int, ...] = (24, 168),
    add_lags: bool = True,
    lags: tuple[int, ...] = (1, 24, 168),
    add_spectral: bool = True,
) -> pd.DataFrame:
    """Convenience pipeline: cyclic + rolling + lag + (optional) spectral features.

    Returns a :class:`pd.DataFrame` regardless of input type.
    """
    from .spectral import fft_features, spectral_entropy
    from .temporal import cyclical_encode, lag_features, rolling_statistics

    out = df.copy()

    if add_cyclic:
        if isinstance(df.index, pd.DatetimeIndex):
            out = cyclical_encode(out)
        else:
            datetime_cols = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
            if datetime_cols:
                out = cyclical_encode(out, datetime_col=datetime_cols[0])
            else:
                out = cyclical_encode(out)

    if add_rolling:
        out = rolling_statistics(out, columns=[target], windows=rolling_windows)

    if add_lags:
        out = lag_features(out, columns=[target], lags=lags, drop_na=False)

    if add_spectral:
        target_arr = out[target].to_numpy()
        spec = fft_features(target_arr, top_k=5)
        ent = spectral_entropy(target_arr)
        for k, v in spec.items():
            out[f"fft_{k}"] = float(v)
        out["spectral_entropy"] = float(ent)
    return out
