from __future__ import annotations

import numpy as np
import pandas as pd

from felits.feature_extraction.automated import extract_all_features, fats_extract, tsfresh_extract


def test_fats_extract_basic() -> None:
    rng = np.random.default_rng(0)
    s = rng.standard_normal(1000)
    feats = fats_extract(s)
    for key in (
        "amplitude",
        "beyond_1_std",
        "beyond_2_std",
        "car_sigma",
        "car_mean",
        "car_std",
        "skew",
        "kurtosis",
        "median_abs_dev",
    ):
        assert key in feats
    assert feats["amplitude"] >= 0


def test_fats_extract_short_returns_nan() -> None:
    feats = fats_extract(np.array([1.0, 2.0]))
    assert np.isnan(feats["amplitude"])


def test_tsfresh_extract_minimal_pipeline() -> None:
    rng = np.random.default_rng(0)
    n_id, n_t = 5, 100
    rows = []
    for i in range(n_id):
        slope = 0.1 * (i + 1)
        for t in range(n_t):
            rows.append({"id": i, "t": t, "v": slope * t + 0.1 * rng.standard_normal()})
    df = pd.DataFrame(rows)
    out = tsfresh_extract(df, column_id="id", column_sort="t")
    assert out.shape[0] == n_id
    assert out.shape[1] > 0


def test_extract_all_features_includes_components(hourly_demand: pd.DataFrame) -> None:
    out = extract_all_features(hourly_demand, target="demand")
    assert isinstance(out, pd.DataFrame)
    assert "hour_sin" in out.columns
    assert "demand_roll24_mean" in out.columns
    assert "demand_lag1" in out.columns
    assert "fft_freq_1" in out.columns
    assert "spectral_entropy" in out.columns
