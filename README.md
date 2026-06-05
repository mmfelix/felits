# FELITS: Feature Engineering and Large-scale Integration for Time Series

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11|3.12|3.13-blue.svg)](https://www.python.org/downloads/)

FELITS is an open-source Python library for end-to-end **time series analysis and forecasting**, with a focus on **Short-Term Load Forecasting (STLF)**. It provides a complete pipeline: signal cleaning, feature engineering, feature selection, predictive modelling, and explainable AI (XAI).

The library is the result of the INIC01-6 research project (CONACYT, Paraguay) and was used to produce the methodology for a published research article.

## Highlights

- **Preprocessing** — outlier detection (IQR, Hampel/MAD, 3-sigma), STL decomposition, dual-scaler pattern
- **Feature extraction** — cyclic encodings, lag/shift features, rolling statistics, FFT, wavelets, tsfresh, FATS-style
- **Feature selection** — Granger causality, KSG mutual information, mRMR, Adaptive LASSO, RF/XGB importance
- **Models** — XGBoost, RandomForest, LinearRegression, plus `tf.keras` RNN models (LSTM/GRU/BiLSTM/BiGRU) and Bahdanau attention variants
- **XAI** — LIME, SHAP, and **Deep SHAP as a closed-loop meta-optimizer** for feature elimination
- **Optuna** hyperparameter optimization with multi-objective TPE sampler
- **Dual API** — all modules accept both `pandas` and `polars` DataFrames; internal logic uses `polars` for performance

## Requirements

- Python ≥3.11, <3.14 (TensorFlow requires ≤3.13)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
git clone https://github.com/felits/felits.git
cd felits

# Create venv and install with uv (recommended)
uv venv --python 3.13
uv pip install -e ".[all,dev]"
```

### Extra dependency groups

| Extra | Includes |
|---|---|
| `[dl]` | TensorFlow for RNN/Attention models |
| `[xgb]` | XGBoost |
| `[wavelet]` | PyWavelets |
| `[duckdb]` | DuckDB for SQL-style batch feature engineering |
| `[all]` | Everything above |

### Python version management

uv handles Python downloads automatically. To switch Python versions:

```bash
uv python install 3.13       # download Python 3.13
uv venv --python 3.13 .venv  # create a venv with it
```

## Quickstart

```python
import polars as pl
from felits.preprocessing import HampelFilter, TimeSeriesScaler
from felits.feature_extraction import cyclical_encode, rolling_statistics
from felits.feature_selection import FeatureSelector
from felits.models import XGBoostForecaster
from felits import Metrics

df = pl.read_csv("demand.csv", try_parse_dates=True)
df = df.with_columns(
    pl.Series("value_clean", HampelFilter(window_size=24).transform(df["value"]))
)
df = cyclical_encode(df, datetime_col="timestamp")
df = rolling_statistics(df, columns=["value_clean"], windows=[24, 168], stats=["mean", "std"])

model = XGBoostForecaster(n_estimators=500, max_depth=6)
model.fit(X_train, y_train)
preds = model.predict(X_test)

m = Metrics(y_test, preds)
print(m.dict_metrics())
```

## Project layout

```
felits/
├── _compat.py               # pandas/polars compatibility layer
├── preprocessing/           # outliers, decomposition, scaling, imputation, metrics
├── feature_extraction/      # temporal, spectral, automated
├── feature_selection/       # causal, information, regularization, ensemble, xai
├── models/                  # base, sklearn, dl (separate modules)
│   ├── base.py              #   _SklearnForecaster, TF detection
│   ├── sklearn.py           #   XGBoost/RF/Linear wrappers
│   └── dl.py                #   RNN/Attention models (requires TF)
├── optimization.py          # Optuna wrappers
├── xai.py                   # LIME, SHAP, deep SHAP closed-loop
└── data.py                  # loaders and synthetic data
examples/                    # Runnable .py example scripts
notebooks/                   # Jupyter notebooks for each example
tests/                       # pytest suite (79+ tests)
```

## Citation

If you use FELITS in academic work, please cite the associated research article (see `research-docs/`).

## License

MIT — see `LICENSE`.
