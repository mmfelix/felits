"""Shared pytest fixtures for the FELITS test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_demand() -> pd.DataFrame:
    """Synthetic hourly electricity-demand series with daily + weekly cycles."""
    rng = np.random.default_rng(42)
    n = 24 * 30  # 30 days
    t = np.arange(n)
    daily = 100 * np.sin(2 * np.pi * t / 24 - np.pi / 2) + 500
    weekly = 30 * np.sin(2 * np.pi * t / (24 * 7))
    noise = rng.normal(0, 5, size=n)
    values = daily + weekly + noise
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"demand": values}, index=idx)


@pytest.fixture
def hourly_demand_with_outliers(hourly_demand: pd.DataFrame) -> pd.DataFrame:
    df = hourly_demand.copy()
    df.iloc[10, 0] = 9999.0
    df.iloc[100, 0] = -500.0
    return df


@pytest.fixture
def synthetic_classification() -> tuple[np.ndarray, np.ndarray]:
    """Tiny dataset for feature-selection tests."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] + 0.5 * X[:, 3] ** 2 - 0.3 * X[:, 7] + rng.normal(scale=0.1, size=200) > 0).astype(
        int
    )
    return X, y
