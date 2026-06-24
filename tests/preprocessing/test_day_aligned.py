"""Tests for the day-aligned windowing and DST-aware grid cleaning.

These tests lock the contract that:

  1. ``SlidingWindowSplitter(day_aligned=True)`` anchors every target to a
     calendar-day boundary at 00:00, regardless of ``HIST_WINDOW``.
  2. The cleaned hourly grid used by the training pipeline is contiguous,
     has 24 rows per day, starts at 00:00, and has no duplicates.

Together they guarantee the user-facing invariant::

    Y = {X0, X1, ..., X23}   where X0 is always the prediction at hour 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from felits.preprocessing import SlidingWindowSplitter

# Make the training scripts importable so we can exercise the real
# ``_clean_hourly_grid`` used by ``prepare_data.py``.
TRAINING_DIR = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "paper"
    / "scripts"
    / "training"
)
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


# ---------------------------------------------------------------------------
# SlidingWindowSplitter(day_aligned=True)
# ---------------------------------------------------------------------------


def _build_clean_grid(n_days: int) -> pd.DataFrame:
    """Build a clean hourly grid (no DST gaps) for ``n_days`` days."""
    idx = pd.date_range("2024-01-01 00:00", periods=n_days * 24, freq="h")
    return pd.DataFrame(
        {
            "f1": np.arange(len(idx), dtype=float),
            "SIN": np.arange(len(idx), dtype=float),
        },
        index=idx,
    )


@pytest.mark.parametrize("hist", [24, 48, 300, 336])
def test_day_aligned_y_covers_hours_0_to_23(hist: int) -> None:
    """For any HIST_WINDOW, every target covers hours 0..23 of a calendar day.

    This is the core invariant: the prediction is always a full calendar day
    with X0 = hour 0, regardless of the historical window size.
    """
    df = _build_clean_grid(n_days=15)
    sp = SlidingWindowSplitter(
        target="SIN", hist_window=hist, pred_window=24,
        day_aligned=True, scaling_type=None,
    )
    ws = sp.split(df)

    n = len(df)
    hours = np.asarray(df.index.hour)
    midnight_positions = np.flatnonzero(hours == 0)
    valid = midnight_positions[midnight_positions >= hist]
    valid = valid[valid + 24 <= n]
    expected_n = len(valid)
    assert ws.y.shape == (expected_n, 24)
    assert ws.X.shape == (expected_n, hist, df.shape[1])

    # Each y[k] is the SIN values at the 24 positions starting at midnight m_k.
    for k, m in enumerate(valid):
        expected = df["SIN"].iloc[m:m + 24].to_numpy()
        np.testing.assert_allclose(ws.y[k], expected)


@pytest.mark.parametrize("hist", [24, 48, 300, 336])
def test_day_aligned_first_y_starts_at_midnight(hist: int) -> None:
    """The first y starts at the first 00:00 with >= HIST_WINDOW preceding hours.

    Hours before that are the *offset* (without effect) and do not form any
    target. This matches the user requirement: leading hours are left unused
    so the first Y falls on 00:00.
    """
    df = _build_clean_grid(n_days=15)
    sp = SlidingWindowSplitter(
        target="SIN", hist_window=hist, pred_window=24,
        day_aligned=True, scaling_type=None,
    )
    ws = sp.split(df)
    # The first target row in df (the corresponding 00:00 timestamp).
    hours = np.asarray(df.index.hour)
    midnight_positions = np.flatnonzero(hours == 0)
    valid = midnight_positions[midnight_positions >= hist]
    valid = valid[valid + 24 <= len(df)]
    first_midnight_pos = int(valid[0])
    first_target_ts = df.index[first_midnight_pos]
    # ws.y[0] corresponds to df.iloc[first_midnight_pos : first_midnight_pos+24]
    expected_first_value = df["SIN"].iloc[first_midnight_pos]
    assert ws.y[0][0] == expected_first_value
    # And the first target's timestamp is at 00:00.
    assert first_target_ts.hour == 0
    # Offset = first_midnight_pos - hist (always >= 0; zero when hist % 24 == 0
    # and 0 <= first_midnight_pos < 24, else positive).
    offset = first_midnight_pos - hist
    assert offset >= 0


def test_day_aligned_with_non_multiple_of_24_hist() -> None:
    """With HIST_WINDOW=50 (not a multiple of 24), targets are still 00:00-23:00.

    The legacy positional splitter would produce targets starting at hour 2
    in this case. The day-aligned mode anchors to 00:00 regardless.
    """
    df = _build_clean_grid(n_days=10)
    sp = SlidingWindowSplitter(
        target="SIN", hist_window=50, pred_window=24,
        day_aligned=True, scaling_type=None,
    )
    ws = sp.split(df)
    # The first 00:00 with >= 50 preceding hours is the 3rd midnight (pos 48
    # has only 48 preceding hours, pos 72 has 72 >= 50).
    first_midnight = next(
        i for i, t in enumerate(df.index) if t.hour == 0 and i >= 50
    )
    assert ws.y[0][0] == df["SIN"].iloc[first_midnight]
    assert ws.y[0][-1] == df["SIN"].iloc[first_midnight + 23]
    # Offset = first_midnight - 50 = 72 - 50 = 22
    assert first_midnight - 50 == 22


def test_day_aligned_rejects_non_24_pred_window() -> None:
    """day_aligned=True requires pred_window=24 (one calendar day)."""
    with pytest.raises(ValueError, match=r"pred_window=24"):
        SlidingWindowSplitter(
            target="SIN", hist_window=24, pred_window=12,
            day_aligned=True, scaling_type=None,
        )


def test_day_aligned_legacy_unchanged() -> None:
    """day_aligned=False keeps the legacy positional behavior (regression guard)."""
    df = _build_clean_grid(n_days=5)
    sp_legacy = SlidingWindowSplitter(
        target="SIN", hist_window=24, pred_window=24,
        scaling_type=None,
    )
    ws = sp_legacy.split(df)
    # Legacy positional: n_samples = (120 - 24) // 24 = 4
    assert ws.y.shape == (4, 24)
    # y[0] = data[24:48]
    np.testing.assert_allclose(ws.y[0], df["SIN"].iloc[24:48].to_numpy())


def test_day_aligned_shared_scaler_with_val() -> None:
    """day_aligned works with the pre-fitted scaler reuse (no scaling leakage)."""
    df_train = _build_clean_grid(n_days=10)
    df_val = _build_clean_grid(n_days=5)  # separate, clean
    df_val.index = df_val.index + pd.Timedelta(days=10)

    sp_t = SlidingWindowSplitter(
        target="SIN", hist_window=48, pred_window=24,
        day_aligned=True, scaling_type="standard",
    )
    ws_t = sp_t.split(df_train)
    sp_v = SlidingWindowSplitter(
        target="SIN", hist_window=48, pred_window=24,
        day_aligned=True, scaling_type="standard",
    )
    ws_v = sp_v.split(df_val, scaler=ws_t.scaler)
    assert ws_v.scaler is ws_t.scaler
    assert ws_v.y.shape[1] == 24


# ---------------------------------------------------------------------------
# _clean_hourly_grid
# ---------------------------------------------------------------------------


from prepare_data import _clean_hourly_grid  # noqa: E402


def test_clean_hourly_grid_removes_duplicates() -> None:
    """Duplicated naive timestamps are averaged into a single row."""
    # 1 full day with a duplicated 01:00 (fall-back analog).
    base = pd.date_range("2024-01-01 00:00", periods=24, freq="h")
    # 01:00 appears twice with different values; the others are unique.
    dup_ts = base.delete(1).insert(1, base[1]).insert(2, base[1])
    values_f1 = list(np.arange(24, dtype=float))
    values_f1[1] = 2.0
    values_f1.insert(2, 4.0)  # the duplicate, shifted
    values_sin = [v * 10 for v in values_f1]
    df = pd.DataFrame({"f1": values_f1, "SIN": values_sin}, index=dup_ts)
    out = _clean_hourly_grid(df)
    assert len(out) == 24
    # The duplicated 01:00 is averaged to (2 + 4) / 2 = 3.0
    assert out.loc[base[1], "f1"] == 3.0
    assert out.loc[base[1], "SIN"] == 30.0
    assert not out.index.has_duplicates


def test_clean_hourly_grid_fills_dst_gaps() -> None:
    """Missing hourly rows are filled with ffill (default) on the gap rows only.

    1 full day with a gap at 02:00 (the DST spring-forward analog).
    """
    # Start with all 24 hours, then drop the 02:00 row.
    base = pd.date_range("2024-03-10 00:00", periods=24, freq="h")
    keep = base.delete(2)  # 23 timestamps
    values = np.arange(24, dtype=float)
    keep_values = np.delete(values, 2)  # 23 values matching the 23 timestamps
    df = pd.DataFrame(
        {"f1": keep_values, "SIN": keep_values * 10},
        index=keep,
    )
    out = _clean_hourly_grid(df, fill_method="ffill")
    assert len(out) == 24
    # The filled hour 02:00 takes the value of hour 01:00 (ffill): 1.0 / 10.0
    assert out.loc["2024-03-10 02:00", "f1"] == 1.0
    assert out.loc["2024-03-10 02:00", "SIN"] == 10.0
    # Hour 03:00 was the original value at index 3 of [0,1,3,4,...] = 3.0
    assert out.loc["2024-03-10 03:00", "f1"] == 3.0
    assert out.loc["2024-03-10 04:00", "f1"] == 4.0


def test_clean_hourly_grid_bfill() -> None:
    """bfill fills gaps with the next known value."""
    base = pd.date_range("2024-03-10 00:00", periods=24, freq="h")
    keep = base.delete(2)  # 23 timestamps
    values = np.arange(24, dtype=float)
    keep_values = np.delete(values, 2)
    df = pd.DataFrame({"SIN": keep_values}, index=keep)
    out = _clean_hourly_grid(df, fill_method="bfill")
    # The gap at 02:00 is filled with the value at 03:00 (= 3.0, bfill).
    assert out.loc["2024-03-10 02:00", "SIN"] == 3.0


def test_clean_hourly_grid_strips_timezone() -> None:
    """A tz-aware index is converted to naive local time."""
    ts = pd.date_range("2024-01-01 00:00", periods=24, freq="h", tz="UTC")
    df = pd.DataFrame({"SIN": np.arange(24, dtype=float)}, index=ts)
    out = _clean_hourly_grid(df)
    assert out.index.tz is None
    assert out.index[0].hour == 0


def test_clean_hourly_grid_asserts_invariants() -> None:
    """The final grid must satisfy: starts at 00:00, 24 rows/day, contiguous."""
    # Build a grid that is ALREADY clean to make sure no false positives.
    ts = pd.date_range("2024-01-01 00:00", periods=48, freq="h")
    df = pd.DataFrame({"SIN": np.arange(48, dtype=float)}, index=ts)
    out = _clean_hourly_grid(df)
    assert out.index[0].hour == 0
    counts = out.index.normalize().value_counts()
    assert (counts == 24).all()
    np.testing.assert_array_equal(out.index.hour, np.arange(len(out)) % 24)
