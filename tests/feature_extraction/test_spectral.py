from __future__ import annotations

import numpy as np
import pytest

from felits.feature_extraction.spectral import (
    fft_features,
    spectral_entropy,
    wavelet_features,
    welch_psd,
)


def test_fft_features_picks_dominant_frequency() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(256)
    signal = np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(256)
    feats = fft_features(signal, top_k=3, sampling_rate=1.0)
    assert "freq_1" in feats and "amp_1" in feats
    # The dominant frequency should be close to 1/24 (cycles per sample).
    assert abs(feats["freq_1"] - 1 / 24) < 0.01


def test_fft_features_requires_minimum_size() -> None:
    with pytest.raises(ValueError):
        fft_features(np.array([1.0]))


def test_spectral_entropy_pure_sine_is_low() -> None:
    t = np.arange(1024)
    s = np.sin(2 * np.pi * t / 32)
    h = spectral_entropy(s, sampling_rate=1.0)
    # A pure sine concentrates almost all energy in one bin.
    assert h < 0.5


def test_spectral_entropy_white_noise_is_high() -> None:
    rng = np.random.default_rng(0)
    s = rng.standard_normal(2048)
    h = spectral_entropy(s, sampling_rate=1.0)
    # White noise spreads energy across bins -> high entropy.
    assert h > 0.5


def test_welch_psd_returns_positive_psd() -> None:
    rng = np.random.default_rng(0)
    f, p = welch_psd(rng.standard_normal(512), sampling_rate=1.0)
    assert (p >= 0).all()
    assert f.size == p.size


def test_wavelet_features_import_error() -> None:
    # If PyWavelets is not installed, the function should raise an
    # informative ImportError. If it is installed, return a dict with
    # approximation and detail arrays.
    try:
        import pywt  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            wavelet_features(np.arange(64, dtype=float))
    else:
        feats = wavelet_features(np.arange(64, dtype=float), wavelet="db4", level=2)
        assert "approximation" in feats
        assert "detail_1" in feats
        assert "detail_2" in feats
