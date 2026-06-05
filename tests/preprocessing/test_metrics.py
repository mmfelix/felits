from __future__ import annotations

import numpy as np
import pytest

from felits.preprocessing.metrics import Metrics, mae, mape, mse, r2, rmse, smape


def test_metrics_basic() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    m = Metrics(y_true, y_pred)
    assert m.mse() == pytest.approx(0.025, abs=1e-9)
    assert m.mae() == pytest.approx(0.15, abs=1e-9)
    assert m.rmse() == pytest.approx(0.15811388, abs=1e-6)
    assert m.r2() == pytest.approx(0.98, abs=1e-2)
    assert "MAPE" in m.dict_metrics()
    assert "sMAPE" in m.dict_metrics()


def test_metrics_shape_check() -> None:
    with pytest.raises(ValueError):
        Metrics(np.zeros(3), np.zeros(4))


def test_smape_handles_zeros() -> None:
    yt = np.array([0.0, 1.0])
    yp = np.array([0.0, 2.0])
    val = smape(yt, yp)
    assert 0.0 <= val <= 200.0


def test_mape_with_zero_returns_large_value() -> None:
    # Modern sklearn no longer raises on zero denominators; it returns inf
    # or a very large value. We just make sure the function does not crash.
    val = mape(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert val == float("inf") or val > 1e6


def test_metrics_function_aliases() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert mse(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0
    assert mae(y_true, y_pred) == 0.0
    assert r2(y_true, y_pred) == 1.0
