"""Example 02: Feature extraction (temporal, spectral, tsfresh)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from felits.data import make_synthetic_ts
from felits.feature_extraction import (
    cyclical_encode,
    extract_all_features,
    fats_extract,
    fft_features,
    lag_features,
    rolling_statistics,
    shift_features,
    spectral_entropy,
    tsfresh_extract,
)


def main() -> None:
    df = make_synthetic_ts(n_samples=24 * 60, n_features=2, seed=0)

    # 1) Cyclical encodings
    df = cyclical_encode(df, period=24)
    print("Cyclic columns added:", [c for c in df.columns if "_sin" in c or "_cos" in c][:4], "...")

    # 2) Lag, rolling, shift
    df = lag_features(df, columns=["y"], lags=[1, 24, 168])
    df = rolling_statistics(df, columns=["y"], windows=[24, 168], stats=["mean", "std"])
    df = shift_features(df, columns=["x0"], shifts=[24])  # t+24 for day-ahead forecasting
    print(f"Feature engineering: {df.shape[1]} columns after engineering.")

    # 3) Spectral features
    fft = fft_features(df["y"].dropna().to_numpy(), top_k=3, sampling_rate=1.0)
    print("FFT top-3:", {k: round(v, 4) for k, v in fft.items()})
    h = spectral_entropy(df["y"].dropna().to_numpy(), sampling_rate=1.0)
    print(f"Spectral entropy: {h:.4f}")

    # 4) FATS-style scalar features
    print(
        "FATS features:",
        {k: round(v, 4) for k, v in fats_extract(df["y"].dropna().to_numpy()).items()},
    )

    # 5) Automated extraction via tsfresh
    long = pd.DataFrame(
        {
            "id": np.repeat(np.arange(5), 50),
            "t": np.tile(np.arange(50), 5),
            "v": np.random.default_rng(0).standard_normal(250).cumsum(),
        }
    )
    feats = tsfresh_extract(long, column_id="id", column_sort="t")
    print(f"tsfresh features: {feats.shape}")

    # 6) Convenience pipeline
    out = extract_all_features(
        make_synthetic_ts(n_samples=24 * 30, n_features=0, seed=0),
        target="y",
    )
    print(f"extract_all_features: {out.shape}")


if __name__ == "__main__":
    main()
