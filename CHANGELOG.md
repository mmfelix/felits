# Changelog

All notable changes to FELITS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] — 2026-06-11

### Changed

- **BREAKING:** Removed `felits/_compat.py` and the entire dual pandas/polars
  compatibility layer. All public APIs now accept and return `pd.DataFrame` /
  `pd.Series` exclusively. If your code imports from `felits._compat`, you must
  migrate to direct pandas usage.
- All public functions in `felits.feature_extraction.temporal` (`cyclical_encode`,
  `lag_features`, `rolling_statistics`, `shift_features`) now operate natively on
  `pd.DataFrame` and return `pd.DataFrame`. Rolling statistics use
  `pd.Series.rolling()` instead of polars expression chains.
- `tsfresh_extract` and `extract_all_features` in `felits.feature_extraction.automated`
  now accept `pd.DataFrame` and return `pd.DataFrame` directly, eliminating the
  internal polars-to-pandas round-trip.
- `TimeSeriesScaler` and `SlidingWindowSplitter` in `felits.preprocessing.scaling`
  now accept `pd.DataFrame` instead of `Union[pd.DataFrame, pl.DataFrame]`.
  The `DataFrameLike` type alias has been removed.
- `forward_fill`, `linear_interpolate`, and `time_aware_interpolate` in
  `felits.preprocessing.imputation` now use pandas native `ffill()` and
  `interpolate()` methods. `time_aware_interpolate` uses `method="time"` for
  `DatetimeIndex` inputs and falls back to `method="linear"` otherwise.
- `DecompositionResult` type annotations changed from
  `pd.Series | pl.Series | np.ndarray` to `pd.Series | np.ndarray`.
- `_back_to_native` in `felits.preprocessing.decomposition` simplified to handle
  only `pd.Series` and `np.ndarray` (no more polars `Series` branch).
- `seasonal_adjust` now preserves the input type: returns `pd.Series` when given
  a `pd.Series`, `np.ndarray` when given an `np.ndarray`.
- `ArrayLike` in `felits.preprocessing.outliers` narrowed from
  `Union[np.ndarray, pl.Series, pl.DataFrame, pd.Series, pd.DataFrame]` to
  `Union[np.ndarray, pd.Series, pd.DataFrame]`.
- `_as_1d_float64` simplified to direct `isinstance` checks against pandas types.
- `pandas` version constraint tightened from `>=2.0` to `>=2.2,<3` to ensure
  compatibility with the new interpolation and rolling APIs while avoiding
  pandas 3.0 Copy-on-Write breaking changes.
- Improved temporal statistics helper with dictionary-based dispatch.

### Removed

- **`polars`** dependency (~50 MB wheel including `polars-runtime-32`).
- `felits._compat` module: `to_polars`, `to_pandas`, `is_polars`, `is_pandas`,
  `to_numpy`, `with_columns`, `has_datetime_column`, `datetime_columns`,
  `is_pandas_datetime_index`, `DataFrameLike`, `SeriesLike`.
- All `import polars as pl` statements across the library and test suite.
- All `isinstance(out, pl.DataFrame)` assertions in tests (replaced with
  `isinstance(out, pd.DataFrame)`).
- `Union[pd.DataFrame, pl.DataFrame]` type signatures in scaling module.

### Fixed

- Test assertions using polars-style positional indexing (`out["col"][0]`) now
  use pandas `.iloc[0]` for correct label-based indexing.
- Rolling null checks in tests now use `np.isnan()` instead of `is None` to
  match pandas `NaN` semantics.
- `time_aware_interpolate` now uses pandas `method="time"` for `DatetimeIndex`
  inputs, providing true time-weighted interpolation instead of linear fallback.
- `spectral.py` `_as_1d` helper now returns a writable copy to avoid downstream
  `ValueError` on read-only numpy arrays.

### Migration guide (0.2.x → 0.3.0)

If you were using the polars-compatible API:

```python
# Before (polars)
import polars as pl
from felits._compat import to_polars

df = pl.read_csv("data.csv", try_parse_dates=True)
result = cyclical_encode(df, datetime_col="timestamp")

# After (pandas)
import pandas as pd

df = pd.read_csv("data.csv", parse_dates=["timestamp"], index_col="timestamp")
result = cyclical_encode(df)
```

All function signatures remain identical except that `pl.DataFrame` inputs are
no longer accepted. Pass `pd.DataFrame` instead.

## [0.2.0] — 2026-06-05

### Added

- **New models:** `LSTMAttentionForecaster`, `PatchTSTForecaster`,
  `LightGBMForecaster` with full sklearn-compatible API.
- `RNNAttentionModel` base class for custom attention-based RNN architectures.
- **Metrics module** expanded: `bias`, `smape`, `max_error` added to
  `felits.preprocessing.metrics`.
- **Feature selection** pipeline overhaul: `FeatureSelector` now supports
  chained selection strategies via `felits.feature_selection.pipeline`.
- **Ensemble feature selection** via `felits.feature_selection.ensemble` with
  voting and union/aggregation across multiple selectors.
- **Deep SHAP closed-loop** meta-optimizer for iterative feature elimination
  in `felits.xai`.
- Synthetic data generator in `felits.data` for quick prototyping.
- Strict typing across the entire codebase (`mypy` clean).
- NumPy-style docstrings on all public classes and functions.

### Changed

- `XGBoostForecaster` and `RandomForestForecaster` refactored to share
  `_SklearnForecaster` base with `LinearForecaster`.
- `RNNBasedModel` refactored: cleaner TF detection, modular layer building.
- `TimeSeriesScaler` and `SlidingWindowSplitter` significantly expanded
  with dual-scaler pattern, jump/overlap modes, and `WindowedSplit` dataclass.
- Feature selection `xai.py` rewritten with cleaner SHAP/LIME integration.
- Scalers now accept `Union[pd.DataFrame, pl.DataFrame]` via `DataFrameLike`.

### Fixed

- `_SklearnForecaster` added to `felits.models.__init__.__all__`.
- Formatting pass applied to 7 files via `ruff format`.

## [0.1.0] — 2026-06-04

### Added

- Initial public release.
- **Preprocessing:** `HampelFilter`, `iqr_outlier_detection`, `three_sigma_filter`,
  `stl_decompose`, `seasonal_adjust`, `TimeSeriesScaler`, `SlidingWindowSplitter`,
  `forward_fill`, `linear_interpolate`, `time_aware_interpolate`, `Metrics`.
- **Feature extraction:** `cyclical_encode`, `lag_features`, `rolling_statistics`,
  `shift_features`, `fft_features`, `spectral_entropy`, `wavelet_features`,
  `tsfresh_extract`, `fats_extract`, `extract_all_features`.
- **Feature selection:** `granger_feature_selection`, `mutual_information_ksg`,
  `mrmr_selection`, `lasso_selection`, `adaptive_lasso_selection`,
  `elastic_net_selection`, `rf_importance_selection`,
  `xgboost_importance_selection`, `permutation_importance_selection`,
  `FeatureSelector`, `select_features`.
- **Models:** `XGBoostForecaster`, `RandomForestForecaster`,
  `LinearForecaster`, `RNNBasedModel` (LSTM/GRU/BiLSTM/BiGRU variants).
- **XAI:** `shap_feature_selection`, `shap_interaction_selection`,
  `lime_explain_instance`, `explain_forecast`, `deep_shap_selector`.
- **Optimization:** `OptunaOptimizer` with multi-objective TPE sampler.
- Dual pandas/polars API via `felits._compat` compatibility layer.
- DuckDB optional extra for SQL-style batch feature engineering.
- CI/CD with GitHub Actions (lint, test, publish to PyPI on tags).
- Runnable example scripts and Jupyter notebooks.

[Unreleased]: https://github.com/mmfelix/felits/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/mmfelix/felits/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/mmfelix/felits/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mmfelix/felits/releases/tag/v0.1.0
