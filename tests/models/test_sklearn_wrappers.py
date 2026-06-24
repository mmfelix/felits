"""Tests for sklearn-compatible forecasters and deep-learning models."""

from __future__ import annotations

import numpy as np
import pytest

from felits.models import (
    LinearForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
    is_dl_available,
)


def _toy_dataset(
    n: int = 200, t: int = 24, f: int = 1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a toy 3D time-series dataset for testing.

    Parameters
    ----------
    n : int
        Number of samples.
    t : int
        Number of time steps.
    f : int
        Number of features.
    seed : int
        Random seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix (n, t, f) and target matrix (n, 5).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, t, f)).astype("float32")
    y = rng.standard_normal((n, 5)).astype("float32")
    return X, y


def test_random_forest_forecaster_flattens_3d() -> None:
    X, y = _toy_dataset()
    model = RandomForestForecaster(n_estimators=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_linear_forecaster() -> None:
    X, y = _toy_dataset()
    model = LinearForecaster()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


@pytest.mark.xgb
def test_xgboost_forecaster() -> None:
    X, y = _toy_dataset()
    model = XGBoostForecaster(n_estimators=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


@pytest.mark.xgb
def test_xgboost_forecaster_device_cpu() -> None:
    X, y = _toy_dataset()
    model = XGBoostForecaster(n_estimators=20, random_state=0, device="cpu")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


@pytest.mark.xgb
def test_xgboost_forecaster_params_passthrough() -> None:
    X, y = _toy_dataset()
    model = XGBoostForecaster(
        n_estimators=20,
        random_state=0,
        device="cpu",
        subsample=0.5,
        colsample_bytree=0.5,
        max_bin=64,
        min_child_weight=3,
        reg_lambda=2.0,
        reg_alpha=0.5,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_predict_before_fit_raises() -> None:
    model = RandomForestForecaster()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.zeros((10, 5)))


def test_is_dl_available() -> None:
    assert isinstance(is_dl_available(), bool)


@pytest.mark.dl
@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_rnn_based_model_compiles() -> None:
    from felits.models import RNNBasedModel

    X, y = _toy_dataset(n=32, t=12, f=1)
    model = RNNBasedModel(
        model_type="LSTM", timesteps=12, features=1, num_units=16, dropout=0.0, output_units=5
    )
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.fit(X, y, epochs=1, batch_size=16, verbose=0, shuffle=False)
    preds = model.predict(X, verbose=0)
    assert preds.shape == y.shape


@pytest.mark.dl
@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_rnn_attention_model_compiles() -> None:
    from felits.models import RNNAttentionModel

    X, y = _toy_dataset(n=32, t=12, f=1)
    model = RNNAttentionModel(
        model_type="GRU", timesteps=12, features=1, num_units=16, dropout=0.0, output_units=5
    )
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.fit(X, y, epochs=1, batch_size=16, verbose=0, shuffle=False)
    preds = model.predict(X, verbose=0)
    assert preds.shape == y.shape


@pytest.mark.dl
@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_lstm_attention_forecaster_compiles() -> None:
    from felits.models import LSTMAttentionForecaster

    X, y = _toy_dataset(n=32, t=12, f=1)
    model = LSTMAttentionForecaster(
        timesteps=12, features=1, lstm_units=16, num_heads=2, output_units=5
    )
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.fit(X, y, epochs=1, batch_size=16, verbose=0, shuffle=False)
    preds = model.predict(X, verbose=0)
    assert preds.shape == y.shape


@pytest.mark.dl
@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_patchtst_forecaster_compiles() -> None:
    from felits.models import PatchTSTForecaster

    X, y = _toy_dataset(n=32, t=48, f=1)
    model = PatchTSTForecaster(
        timesteps=48,
        features=1,
        patch_length=8,
        patch_stride=4,
        d_model=32,
        num_heads=2,
        ff_dim=64,
        num_layers=1,
        output_units=5,
    )
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.fit(X, y, epochs=1, batch_size=16, verbose=0, shuffle=False)
    preds = model.predict(X, verbose=0)
    assert preds.shape == y.shape


@pytest.mark.dl
@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_bahdanau_attention_shape() -> None:
    import tensorflow as tf

    from felits.models import BahdanauAttention

    layer = BahdanauAttention()
    x = tf.random.normal((4, 10, 8))
    out = layer(x)
    assert out.shape == (4, 8)
