from __future__ import annotations

import numpy as np
import pytest

from felits.models import (
    LinearForecaster,
    RandomForestForecaster,
    is_dl_available,
)


def _toy_dataset(
    n: int = 200, t: int = 24, f: int = 1, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
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


def test_is_dl_available() -> None:
    assert isinstance(is_dl_available(), bool)


@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_rnn_based_model_compiles() -> None:
    from felits.models import RNNBasedModel

    X, y = _toy_dataset(n=32, t=12, f=1)
    model = RNNBasedModel(
        type="LSTM", timesteps=12, features=1, num_units=16, dropout=0.0, output_units=5
    )
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.fit(X, y, epochs=1, batch_size=16, verbose=0, shuffle=False)
    preds = model.predict(X, verbose=0)
    assert preds.shape == y.shape


@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_rnn_attention_model_compiles() -> None:
    from felits.models import RNNAttentionModel

    X, y = _toy_dataset(n=32, t=12, f=1)
    model = RNNAttentionModel(
        type="GRU", timesteps=12, features=1, num_units=16, dropout=0.0, output_units=5
    )
    import tensorflow as tf

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    model.fit(X, y, epochs=1, batch_size=16, verbose=0, shuffle=False)
    preds = model.predict(X, verbose=0)
    assert preds.shape == y.shape


@pytest.mark.skipif(not is_dl_available(), reason="TensorFlow not installed")
def test_bahdanau_attention_shape() -> None:
    import tensorflow as tf

    from felits.models import BahdanauAttention

    layer = BahdanauAttention()
    x = tf.random.normal((4, 10, 8))
    out = layer(x)
    assert out.shape == (4, 8)
