"""Data loaders and synthetic time-series generators.

The module exposes:

- :func:`load_example_dataset` — small CSV datasets bundled with FELITS
  for demos and unit tests.
- :func:`make_synthetic_ts` — reproducible synthetic univariate time
  series with optional seasonality / noise.
- :func:`load_sin_data` — convenience loader for the Paraguay SIN
  dataset, used in the original research article. The function
  gracefully returns ``None`` when the file is not present so the test
  suite is not bound to a particular machine.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

__all__ = ["load_example_dataset", "load_sin_data", "make_synthetic_ts"]


def make_synthetic_ts(
    n_samples: int = 1000,
    n_features: int = 1,
    seasonality: bool = True,
    period: int = 24,
    noise_std: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic time series with daily seasonality.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of rows in the generated DataFrame.
    n_features : int, default=1
        Number of exogenous feature columns (named ``"x0"``, ``"x1"``, …).
    seasonality : bool, default=True
        If True, add a sinusoidal seasonal component to the target column.
    period : int, default=24
        Period of the primary seasonal component (in time steps).
    noise_std : float, default=0.1
        Standard deviation of the additive Gaussian noise.
    seed : int, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex (hourly), a ``"y"`` target column,
        and ``n_features`` exogenous columns.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    y = np.zeros(n_samples)
    if seasonality:
        y = 10 * np.sin(2 * np.pi * t / period) + 5 * np.cos(2 * np.pi * t / (period * 7))
    y += noise_std * rng.standard_normal(n_samples)
    df = pd.DataFrame({"y": y}, index=pd.date_range("2024-01-01", periods=n_samples, freq="h"))
    for i in range(n_features):
        df[f"x{i}"] = rng.standard_normal(n_samples)
    return df


def load_example_dataset(name: str = "synthetic_demand") -> pd.DataFrame:
    """Load one of the small example datasets bundled with FELITS.

    The default ``"synthetic_demand"`` is generated on the fly by
    :func:`make_synthetic_ts` so users can run the quickstart without
    any external files.

    Parameters
    ----------
    name : str, default="synthetic_demand"
        Name of the dataset to load.

    Returns
    -------
    pd.DataFrame
        The requested dataset as a pandas DataFrame.

    Raises
    ------
    ValueError
        If the dataset name is not recognized.
    """
    if name == "synthetic_demand":
        return make_synthetic_ts(n_samples=24 * 30)
    raise ValueError(f"Unknown example dataset {name!r}.")


def load_sin_data(path: str | None = None) -> pd.DataFrame | None:
    """Load the Paraguay SIN electricity-demand dataset.

    The function looks for a CSV file at ``path`` or, when ``path`` is
    ``None``, in the conventional ``dataset/processed_dataset.csv``
    location relative to the current working directory. When no file is
    found it returns ``None`` rather than raising, so that the test suite
    is environment-agnostic.

    Parameters
    ----------
    path : str or None, default=None
        Path to the CSV file. If None, defaults to
        ``dataset/processed_dataset.csv``.

    Returns
    -------
    pd.DataFrame or None
        The dataset with a DatetimeIndex, or None if the file is not found.
    """
    if path is None:
        path = os.path.join("dataset", "processed_dataset.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    return df
