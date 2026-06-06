"""Deep-learning models for time-series forecasting (requires TensorFlow)."""

from __future__ import annotations

from typing import Any, Dict

from .base import is_dl_available


def _load_dl() -> Dict[str, Any]:
    """Load TensorFlow Keras modules dynamically.

    Returns
    -------
    dict
        A dictionary containing references to Keras backend, layers, and Model.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    """
    if not is_dl_available():
        raise ImportError(
            "Deep-learning models require TensorFlow. Install with: pip install 'felits[dl]'"
        )
    from tensorflow.keras import backend
    from tensorflow.keras.layers import (
        GRU,
        LSTM,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        Input,
        Layer,
        LayerNormalization,
        MultiHeadAttention,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import plot_model

    return {
        "K": backend,
        "Conv1D": Conv1D,
        "Flatten": Flatten,
        "GRU": GRU,
        "LSTM": LSTM,
        "Bidirectional": Bidirectional,
        "Dense": Dense,
        "Dropout": Dropout,
        "Input": Input,
        "Layer": Layer,
        "MultiHeadAttention": MultiHeadAttention,
        "LayerNormalization": LayerNormalization,
        "Model": Model,
        "plot_model": plot_model,
    }


def _build_rnn(model_type: str, num_units: int, with_attention: bool) -> Any:
    """Build an RNN layer based on the specified type.

    Parameters
    ----------
    model_type : str
        The type of RNN layer to build. Must be one of 'LSTM', 'GRU',
        'BiLSTM', or 'BiGRU'.
    num_units : int
        Number of units in the RNN layer.
    with_attention : bool
        If True, the layer will return sequences (required for attention).

    Returns
    -------
    Any
        A configured Keras RNN layer instance.

    Raises
    ------
    ValueError
        If an unknown `model_type` is provided.
    """
    sym = _load_dl()
    LSTM, GRU, Bidirectional = sym["LSTM"], sym["GRU"], sym["Bidirectional"]

    if model_type == "LSTM":
        return LSTM(units=num_units, return_sequences=with_attention)
    if model_type == "GRU":
        return GRU(units=num_units, return_sequences=with_attention)
    if model_type == "BiLSTM":
        return Bidirectional(LSTM(units=num_units, return_sequences=with_attention))
    if model_type == "BiGRU":
        return Bidirectional(GRU(units=num_units, return_sequences=with_attention))

    raise ValueError(f"Unknown RNN type {model_type!r}; choose from LSTM, GRU, BiLSTM, or BiGRU")


class BahdanauAttention:
    """Bahdanau (additive) attention layer operating on an RNN output sequence.

    This class acts as a factory; the actual ``tf.keras.layers.Layer``
    subclass is built on first instantiation so the module is importable
    without TensorFlow.
    """

    def __new__(cls) -> Any:
        if not is_dl_available():
            raise ImportError(
                "BahdanauAttention requires TensorFlow. Install with: pip install 'felits[dl]'"
            )
        sym = _load_dl()
        backend = sym["K"]
        Layer = sym["Layer"]

        class BahdanauAttentionLayer(Layer):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                super().__init__()

            def build(self, input_shape: tuple) -> None:
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

            def call(self, x: Any) -> Any:
                e = backend.tanh(backend.dot(x, self.W) + self.b)
                e = backend.squeeze(e, axis=-1)
                alpha = backend.softmax(e)
                alpha = backend.expand_dims(alpha, axis=-1)
                context = x * alpha
                context = backend.sum(context, axis=1)
                return context

        return BahdanauAttentionLayer()


def RNNBasedModel(
    model_type: str = "LSTM",
    timesteps: int = 24,
    features: int = 1,
    num_units: int = 64,
    dropout: float = 0.0,
    output_units: int = 24,
    name: str = "RNNBasedModel",
) -> Any:
    """Build a vanilla RNN forecaster (LSTM/GRU/BiLSTM/BiGRU).

    Parameters
    ----------
    model_type : str, default="LSTM"
        The type of RNN to use ('LSTM', 'GRU', 'BiLSTM', 'BiGRU').
    timesteps : int, default=24
        Number of time steps in the input sequence.
    features : int, default=1
        Number of features per time step.
    num_units : int, default=64
        Number of units in the RNN layer.
    dropout : float, default=0.0
        Dropout rate applied after the RNN layer.
    output_units : int, default=24
        Number of units in the output dense layer.
    name : str, default="RNNBasedModel"
        Name of the Keras model.

    Returns
    -------
    Any
        A compiled Keras Model instance.
    """
    if not is_dl_available():
        raise ImportError("RNNBasedModel requires TensorFlow.")

    sym = _load_dl()
    Model = sym["Model"]
    Dropout = sym["Dropout"]
    Dense = sym["Dense"]

    class RNNBasedModelClass(Model):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(name=name)
            self.input_shape_model = (timesteps, features)
            self.dropout_rate = dropout
            self.hidden_layer = _build_rnn(model_type, num_units, with_attention=False)
            self.dropout = Dropout(rate=dropout) if dropout > 0 else None
            self.output_layer = Dense(units=output_units, activation="linear")

        def call(self, inputs: Any) -> Any:
            x = self.hidden_layer(inputs)
            if self.dropout is not None:
                x = self.dropout(x)
            return self.output_layer(x)

    return RNNBasedModelClass()


def RNNAttentionModel(
    model_type: str = "LSTM",
    timesteps: int = 24,
    features: int = 1,
    num_units: int = 64,
    dropout: float = 0.0,
    output_units: int = 24,
    name: str = "RNNAttentionModel",
) -> Any:
    """RNN forecaster with a Bahdanau attention head.

    Parameters
    ----------
    model_type : str, default="LSTM"
        The type of RNN to use ('LSTM', 'GRU', 'BiLSTM', 'BiGRU').
    timesteps : int, default=24
        Number of time steps in the input sequence.
    features : int, default=1
        Number of features per time step.
    num_units : int, default=64
        Number of units in the RNN layer.
    dropout : float, default=0.0
        Dropout rate applied after the RNN layer.
    output_units : int, default=24
        Number of units in the output dense layer.
    name : str, default="RNNAttentionModel"
        Name of the Keras model.

    Returns
    -------
    Any
        A compiled Keras Model instance with attention.
    """
    if not is_dl_available():
        raise ImportError("RNNAttentionModel requires TensorFlow.")

    sym = _load_dl()
    Model = sym["Model"]
    Dropout = sym["Dropout"]
    Dense = sym["Dense"]

    class RNNAttentionModelClass(Model):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(name=name)
            self.input_shape_model = (timesteps, features)
            self.dropout_rate = dropout
            self.hidden_layer = _build_rnn(model_type, num_units, with_attention=True)
            self.dropout = Dropout(rate=dropout) if dropout > 0 else None
            self.attention = BahdanauAttention()
            self.output_layer = Dense(units=output_units, activation="linear")

        def call(self, inputs: Any) -> Any:
            x = self.hidden_layer(inputs)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.attention(x)  # type: ignore[operator]
            return self.output_layer(x)

    return RNNAttentionModelClass()


def LSTMAttentionForecaster(
    timesteps: int = 24,
    features: int = 1,
    lstm_units: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    dropout: float = 0.1,
    output_units: int = 24,
    name: str = "LSTMAttentionForecaster",
) -> Any:
    """Hybrid LSTM + Multi-Head Self-Attention forecaster.

    This architecture combines the local temporal modeling capabilities of LSTM
    with the global dependency capturing of Multi-Head Self-Attention. The LSTM
    processes the sequence and returns all hidden states, which are then fed
    into a self-attention layer to allow the model to focus on the most relevant
    past time steps regardless of their distance.

    Parameters
    ----------
    timesteps : int, default=24
        Number of time steps in the input sequence.
    features : int, default=1
        Number of features per time step.
    lstm_units : int, default=64
        Number of units in the LSTM layer.
    num_heads : int, default=4
        Number of attention heads in the MultiHeadAttention layer.
    ff_dim : int, default=128
        Dimension of the feed-forward network in the attention block.
    dropout : float, default=0.1
        Dropout rate applied after LSTM and attention layers.
    output_units : int, default=24
        Number of units in the output dense layer (forecast horizon).
    name : str, default="LSTMAttentionForecaster"
        Name of the Keras model.

    Returns
    -------
    Any
        A compiled Keras Model instance.
    """
    if not is_dl_available():
        raise ImportError("LSTMAttentionForecaster requires TensorFlow.")

    sym = _load_dl()
    Model = sym["Model"]
    LSTM = sym["LSTM"]
    MultiHeadAttention = sym["MultiHeadAttention"]
    LayerNormalization = sym["LayerNormalization"]
    Dense = sym["Dense"]
    Dropout = sym["Dropout"]
    Input = sym["Input"]

    class LSTMAttentionModelClass(Model):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(name=name)
            self.input_layer = Input(shape=(timesteps, features))
            self.lstm = LSTM(units=lstm_units, return_sequences=True)
            self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=lstm_units)
            self.norm = LayerNormalization()
            self.dropout = Dropout(rate=dropout)
            self.output_layer = Dense(units=output_units, activation="linear")

        def call(self, inputs: Any, training: bool = False) -> Any:
            x = self.lstm(inputs)
            # Self-attention: query, value, and key are all the LSTM outputs
            attn_out = self.attention(query=x, value=x, key=x)
            # Residual connection and normalization
            x = self.norm(x + attn_out)
            if training:
                x = self.dropout(x)
            return self.output_layer(x)

    return LSTMAttentionModelClass()


def PatchTSTForecaster(
    timesteps: int = 336,
    features: int = 1,
    patch_length: int = 16,
    patch_stride: int = 8,
    d_model: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
    output_units: int = 24,
    name: str = "PatchTSTForecaster",
) -> Any:
    """Patch Time Series Transformer (PatchTST) forecaster.

    This is a streamlined implementation of the PatchTST architecture. It
    segments the time series into subseries-level patches, which are then
    served as input tokens to a standard Transformer Encoder. This patching
    design reduces the sequence length quadratically, allowing the model to
    attend to longer historical windows while retaining local semantic
    information within each patch.

    Note: This implementation processes all channels together for simplicity.
    For strict channel-independence (as in the original paper), the input
    should be processed per-channel with shared weights.

    Parameters
    ----------
    timesteps : int, default=336
        Number of time steps in the input sequence (look-back window).
    features : int, default=1
        Number of features per time step.
    patch_length : int, default=16
        Length of each patch (number of time steps per patch).
    patch_stride : int, default=8
        Stride between consecutive patches.
    d_model : int, default=128
        Dimension of the Transformer model and patch embeddings.
    num_heads : int, default=4
        Number of attention heads in the Transformer Encoder.
    ff_dim : int, default=256
        Dimension of the feed-forward network in the Transformer Encoder.
    num_layers : int, default=3
        Number of Transformer Encoder layers.
    dropout : float, default=0.1
        Dropout rate applied in the Transformer Encoder.
    output_units : int, default=24
        Number of units in the output dense layer (forecast horizon).
    name : str, default="PatchTSTForecaster"
        Name of the Keras model.

    Returns
    -------
    Any
        A compiled Keras Model instance.
    """
    if not is_dl_available():
        raise ImportError("PatchTSTForecaster requires TensorFlow.")

    sym = _load_dl()
    Model = sym["Model"]
    Conv1D = sym["Conv1D"]
    Flatten = sym["Flatten"]
    MultiHeadAttention = sym["MultiHeadAttention"]
    LayerNormalization = sym["LayerNormalization"]
    Dense = sym["Dense"]
    Dropout = sym["Dropout"]

    num_patches = (timesteps - patch_length) // patch_stride + 1

    class PatchTSTModelClass(Model):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            super().__init__(name=name)
            self.num_patches = num_patches

            # Patching via Conv1D (acts as local feature extractor + projection)
            self.patch_conv = Conv1D(
                filters=d_model,
                kernel_size=patch_length,
                strides=patch_stride,
                padding="valid",
                use_bias=False,
            )

            self.pos_encoding = self.add_weight(
                shape=(1, self.num_patches, d_model),
                initializer="random_normal",
                trainable=True,
                name="pos_encoding",
            )

            self.encoder_blocks = []
            for _ in range(num_layers):
                self.encoder_blocks.append(
                    {
                        "mha": MultiHeadAttention(num_heads=num_heads, key_dim=d_model),
                        "norm1": LayerNormalization(),
                        "dense1": Dense(ff_dim, activation="relu"),
                        "dense2": Dense(d_model),
                        "dropout": Dropout(rate=dropout),
                        "norm2": LayerNormalization(),
                    }
                )

            self.flatten = Flatten()
            self.output_layer = Dense(units=output_units, activation="linear")

        def call(self, inputs: Any, training: bool = False) -> Any:
            # Patching and projection: (batch, num_patches, d_model)
            x = self.patch_conv(inputs)

            # Add positional encoding
            x = x + self.pos_encoding

            # Transformer Encoder
            for block in self.encoder_blocks:
                # Self-attention
                attn_out = block["mha"](query=x, value=x, key=x)
                x = block["norm1"](x + attn_out)

                # Feed-forward
                ff_out = block["dense2"](block["dense1"](x))
                if training:
                    ff_out = block["dropout"](ff_out)
                x = block["norm2"](x + ff_out)

            # Forecasting head
            x = self.flatten(x)
            return self.output_layer(x)

    return PatchTSTModelClass()
