# FELITS examples

Each example is available as both a runnable Python script (`.py`) and a
Jupyter notebook (`.ipynb`). The `.py` scripts are the canonical versions;
the notebooks are generated from them.

| # | Topic | Script | Notebook |
|---|-------|--------|----------|
| 01 | Preprocessing (outliers, STL, scaling) | [`01_preprocessing.py`](01_preprocessing.py) | [`notebooks/01_preprocessing.ipynb`](../notebooks/01_preprocessing.ipynb) |
| 02 | Feature extraction (temporal, spectral, tsfresh) | [`02_feature_extraction.py`](02_feature_extraction.py) | [`notebooks/02_feature_extraction.ipynb`](../notebooks/02_feature_extraction.ipynb) |
| 03 | Feature selection (Granger, MI, LASSO, SHAP) | [`03_feature_selection.py`](03_feature_selection.py) | [`notebooks/03_feature_selection.ipynb`](../notebooks/03_feature_selection.ipynb) |
| 04 | Deep-learning models (LSTM, GRU, attention) | [`04_deep_learning_models.py`](04_deep_learning_models.py) | [`notebooks/04_deep_learning_models.ipynb`](../notebooks/04_deep_learning_models.ipynb) |
| 05 | XAI explainability (SHAP, LIME, closed-loop) | [`05_xai_explainability.py`](05_xai_explainability.py) | [`notebooks/05_xai_explainability.ipynb`](../notebooks/05_xai_explainability.ipynb) |
| 06 | Full STLF pipeline | [`06_full_pipeline.py`](06_full_pipeline.py) | [`notebooks/06_full_pipeline.ipynb`](../notebooks/06_full_pipeline.ipynb) |

Run any script with:

```bash
uv run python examples/01_preprocessing.py
```

The deep-learning examples (#04, #06) require the optional `[dl]` extra
and a working TensorFlow installation.
