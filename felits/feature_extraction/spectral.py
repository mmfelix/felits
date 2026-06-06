"""Spectral / frequency-domain feature extraction.

The functions in this module produce low-dimensional summaries of the
frequency content of a time series. They are commonly used as auxiliary
features in load forecasting (daily / weekly periodicities) and in
astronomical time-series analysis (the ``FATS`` library, from which some
of the names below are borrowed).

The optional ``wavelet_features`` function requires ``PyWavelets``; it is
imported lazily so the rest of the module works even when the optional
dependency is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as _signal
from scipy.fft import rfft, rfftfreq

__all__ = [
    "fft_features",
    "spectral_entropy",
    "wavelet_features",
    "welch_psd",
]


def _as_1d(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return np.asarray(x).ravel()
    return np.asarray(x, dtype=float).ravel()


def fft_features(series, top_k: int = 5, sampling_rate: float = 1.0) -> dict[str, float]:
    """Return the top-k dominant frequencies, amplitudes and phases of a series.

    Parameters
    ----------
    series:
        1-D array-like.
    top_k:
        Number of dominant frequencies to return (excluding the DC component).
    sampling_rate:
        Samples per unit time. For hourly data use ``1`` (cycles per hour);
        for 30-min data use ``2``, etc.

    Returns
    -------
    dict
        Keys: ``freq_1`` … ``freq_k``, ``amp_1`` … ``amp_k``, ``phase_1`` …
        ``phase_k``. Frequencies are expressed in the same units as
        ``sampling_rate`` (e.g. cycles/day for hourly data with rate=24).
    """
    arr = _as_1d(series)
    n = arr.size
    if n < 2:
        raise ValueError("`series` must contain at least two samples.")
    arr = arr - arr.mean()
    spectrum = rfft(arr)
    freqs = rfftfreq(n, d=1.0 / sampling_rate)
    amps = np.abs(spectrum) / n
    phases = np.angle(spectrum)
    # Skip DC (index 0).
    order = np.argsort(amps[1:])[::-1][:top_k] + 1
    out: dict[str, float] = {}
    for i, idx in enumerate(order, start=1):
        out[f"freq_{i}"] = float(freqs[idx])
        out[f"amp_{i}"] = float(amps[idx])
        out[f"phase_{i}"] = float(phases[idx])
    return out


def spectral_entropy(series, sampling_rate: float = 1.0, normalize: bool = True) -> float:
    """Return the spectral entropy of ``series``.

    Computes the power spectral density via Welch's method, normalises it to
    a probability distribution, and returns its Shannon entropy in nats (or
    in [0, 1] when ``normalize=True``).
    """
    arr = _as_1d(series)
    if arr.size < 4:
        return 0.0
    _, psd = _signal.welch(arr, fs=sampling_rate, scaling="spectrum")
    psd = psd[psd > 0]
    if psd.size == 0:
        return 0.0
    p = psd / psd.sum()
    h = -float(np.sum(p * np.log(p)))
    if normalize:
        h = h / float(np.log(p.size))
    return h


def welch_psd(
    series,
    sampling_rate: float = 1.0,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch's power spectral density.

    Returns the (frequencies, psd) tuple; the caller is responsible for
    further processing (peak picking, band integration, etc.).
    """
    arr = _as_1d(series)
    f, p = _signal.welch(arr, fs=sampling_rate, nperseg=nperseg)
    return f, p


def wavelet_features(
    series,
    wavelet: str = "db4",
    level: int = 3,
) -> dict[str, np.ndarray]:
    """Compute the discrete wavelet decomposition of ``series``.

    Returns a dict with keys ``approximation`` and ``detail_<i>`` for
    each level. Each value is the array of wavelet coefficients.

    Requires the optional dependency ``PyWavelets``.
    """
    try:
        import pywt
    except ImportError as exc:  # pragma: no cover - exercised only without PyWavelets
        raise ImportError(
            "wavelet_features requires PyWavelets. Install with: pip install 'felits[wavelet]'"
        ) from exc
    arr = _as_1d(series)
    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=level)
    out: dict[str, np.ndarray] = {"approximation": coeffs[0]}
    for i, c in enumerate(coeffs[1:], start=1):
        out[f"detail_{i}"] = c
    return out
