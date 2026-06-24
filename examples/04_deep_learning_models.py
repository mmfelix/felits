"""Example 04: Deep-learning models (LSTM, GRU, attention).

Requires the optional ``[dl]`` extra (TensorFlow >= 2.15).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from felits.models import BahdanauAttention, RNNAttentionModel, RNNBasedModel


def _toy(n: int = 200, t: int = 24, f: int = 1, horizon: int = 12):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, t, f)).astype("float32")
    y = rng.standard_normal((n, horizon)).astype("float32")
    return X, y


def main() -> None:
    tf.keras.utils.set_random_seed(7)
    X, y = _toy()

    # Vanilla LSTM
    print("Training vanilla LSTM...")
    lstm = RNNBasedModel(model_type="LSTM", timesteps=24, features=1, num_units=32, output_units=12)
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X, y, epochs=2, batch_size=32, verbose=0, shuffle=False)
    pred_lstm = lstm.predict(X, verbose=0)
    print(f"  LSTM pred shape: {pred_lstm.shape}")

    # LSTM + Bahdanau attention
    print("Training LSTM + Bahdanau attention...")
    attn = RNNAttentionModel(model_type="LSTM", timesteps=24, features=1, num_units=32, output_units=12)
    attn.compile(optimizer="adam", loss="mse")
    attn.fit(X, y, epochs=2, batch_size=32, verbose=0, shuffle=False)
    pred_attn = attn.predict(X, verbose=0)
    print(f"  Attn pred shape: {pred_attn.shape}")

    # Stand-alone attention layer smoke test
    x = tf.random.normal((4, 10, 8))
    out = BahdanauAttention()(x)
    print(f"  BahdanauAttention output shape: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
