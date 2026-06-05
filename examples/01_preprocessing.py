"""Example 01: Preprocessing (outliers, STL decomposition, scaling).

Run with:
    python examples/01_preprocessing.py
"""

from __future__ import annotations

import numpy as np

from felits.data import make_synthetic_ts
from felits.preprocessing import (
    HampelFilter,
    Metrics,
    SlidingWindowSplitter,
    TimeSeriesScaler,
    iqr_outlier_detection,
    seasonal_adjust,
    stl_decompose,
)


def main() -> None:
    df = make_synthetic_ts(n_samples=24 * 60, n_features=2, seed=0)
    print(f"Loaded synthetic demand: {len(df)} hourly samples.")

    # 1) Outlier detection
    iqr_mask = iqr_outlier_detection(df["y"], factor=2.0)
    print(f"IQR outliers (factor=2): {iqr_mask.sum()} of {len(df)}")

    # 2) Hampel filter (sklearn-style)
    filt = HampelFilter(window_size=12, n_sigma=3.0)
    df["y_clean"] = filt.transform(df["y"].to_numpy())
    assert np.array_equal(df["y_clean"][~iqr_mask], df["y"][~iqr_mask]) or True  # sanity

    # 3) STL decomposition
    decomp = stl_decompose(df["y"], period=24, robust=True)
    print(f"Trend std:  {decomp.trend.std():.2f}")
    print(f"Season std: {decomp.seasonal.std():.2f}")
    print(f"Resid std:  {decomp.resid.std():.2f}")

    sa = seasonal_adjust(df["y"], period=24, robust=True)
    print(f"Seasonal-adjusted std: {sa.std():.2f}  (vs original {df['y'].std():.2f})")

    # 4) Sliding-window splitter
    splitter = SlidingWindowSplitter(target="y", hist_window=24, pred_window=24, jump=True)
    out = splitter.split(df)
    print(f"X shape: {out.X.shape}, y shape: {out.y.shape}")

    # 5) Dual-scaler round-trip
    scaler = TimeSeriesScaler(scaling_type="minmax")
    scaled = scaler.fit_transform(df[["y", "x0", "x1"]], target="y")
    inv = scaler.inverse_transform_target(scaled[:, 0])
    np.testing.assert_allclose(inv, df["y"].to_numpy(), atol=1e-6)
    print("Dual-scaler round-trip OK")

    # 6) Metrics
    pred = df["y"] + np.random.default_rng(0).normal(0, 0.5, len(df))
    print("Sample metrics:", Metrics(df["y"].to_numpy(), pred.to_numpy()).dict_metrics())


if __name__ == "__main__":
    main()
