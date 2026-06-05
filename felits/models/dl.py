"""Deep-learning models for time-series forecasting (requires TensorFlow)."""

from __future__ import annotations

from .base import _build_rnn, _load_dl, is_dl_available


class BahdanauAttention:
    """Bahdanau (additive) attention layer operating on an RNN output sequence.

    The layer is exposed as a class factory; the actual ``tf.keras.layers.Layer``
    subclass is built on first instantiation so the module is importable
    without TensorFlow.
    """

    def __new__(cls):
        if not is_dl_available():
            raise ImportError(
                "BahdanauAttention requires TensorFlow. Install with: pip install 'felits[dl]'"
            )
        sym = _load_dl()
        backend = sym["K"]
        Layer = sym["Layer"]

        class BahdanauAttentionLayer(Layer):
            def __init__(self):
                super().__init__()

            def build(self, input_shape):
                self.W = self.add_weight(
                    name="attention_weight",
                    shape=(input_shape[-1], 1),
                    initializer="random_normal",
                    trainable=True,
                )
                self.b = self.add_weight(
                    name="attention_bias",
                    shape=(input_shape[1], 1),
                    initializer="zeros",
                    trainable=True,
                )
                super().build(input_shape)

            def call(self, x):
                e = backend.tanh(backend.dot(x, self.W) + self.b)
                e = backend.squeeze(e, axis=-1)
                alpha = backend.softmax(e)
                alpha = backend.expand_dims(alpha, axis=-1)
                context = x * alpha
                context = backend.sum(context, axis=1)
                return context

        return BahdanauAttentionLayer()


def RNNBasedModel(
    type: str = "LSTM",
    timesteps: int = 24,
    features: int = 1,
    num_units: int = 64,
    dropout: float = 0.0,
    output_units: int = 24,
    name: str = "RNNBasedModel",
):
    """Build a vanilla RNN forecaster (LSTM/GRU/BiLSTM/BiGRU)."""
    if not is_dl_available():
        raise ImportError("RNNBasedModel requires TensorFlow.")
    sym = _load_dl()
    Model = sym["Model"]
    Dropout = sym["Dropout"]
    Dense = sym["Dense"]

    class RNNBasedModel(Model):
        def __init__(self):
            super().__init__(name=name)
            self.input_shape_model = (timesteps, features)
            self.dropout_rate = dropout
            self.hidden_layer = _build_rnn(type, num_units, with_attention=False)
            if dropout != 0:
                self.dropout = Dropout(rate=dropout)
            self.output_layer = Dense(units=output_units, activation="linear")

        def call(self, inputs):
            x = self.hidden_layer(inputs)
            if self.dropout_rate != 0:
                x = self.dropout(x)
            return self.output_layer(x)

    return RNNBasedModel()


def RNNAttentionModel(
    type: str = "LSTM",
    timesteps: int = 24,
    features: int = 1,
    num_units: int = 64,
    dropout: float = 0.0,
    output_units: int = 24,
    name: str = "RNNAttentionModel",
):
    """RNN forecaster with a Bahdanau attention head."""
    if not is_dl_available():
        raise ImportError("RNNAttentionModel requires TensorFlow.")
    sym = _load_dl()
    Model = sym["Model"]
    Dropout = sym["Dropout"]
    Dense = sym["Dense"]

    class RNNAttentionModel(Model):
        def __init__(self):
            super().__init__(name=name)
            self.input_shape_model = (timesteps, features)
            self.dropout_rate = dropout
            self.hidden_layer = _build_rnn(type, num_units, with_attention=True)
            if dropout != 0:
                self.dropout = Dropout(rate=dropout)
            self.attention = BahdanauAttention()
            self.output_layer = Dense(units=output_units, activation="linear")

        def call(self, inputs):
            x = self.hidden_layer(inputs)
            if self.dropout_rate != 0:
                x = self.dropout(x)
            x = self.attention(x)
            return self.output_layer(x)

    return RNNAttentionModel()
