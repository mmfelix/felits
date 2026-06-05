"""Compatibility layer for dual pandas / polars support.

FELITS accepts both ``pd.DataFrame`` and ``pl.DataFrame`` (or a mix of both)
at all public APIs. Internally the library works with :class:`polars.DataFrame`
for performance. Conversion to pandas happens only at library boundaries
(e.g. ``statsmodels``, ``tsfresh``) that require it.

Usage
-----
>>> from felits._compat import to_polars, to_pandas, DataFrameLike
>>>
>>> def my_public_api(df: DataFrameLike, target: str):
...     pdf = to_polars(df)          # guarantee polars
...     result = pdf.select(pl.col(target))
...     return to_pandas(result)     # back to pandas for the caller
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

DataFrameLike = Union["pd.DataFrame", "pl.DataFrame"]
SeriesLike = Union["pd.Series", "pl.Series"]


def to_polars(data: object, *columns: str, include_index: bool = False) -> "pl.DataFrame":
    """Convert ``data`` to a :class:`polars.DataFrame`.

    Parameters
    ----------
    include_index:
        When ``True`` and ``data`` is a pandas DataFrame with a DatetimeIndex,
        the index is included as the first column named ``"_time"``.
    """
    import polars as pl

    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pl.Series):
        return data.to_frame()
    if isinstance(data, str):
        raise TypeError(f"Cannot convert str to polars DataFrame: {data!r}")

    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if include_index and isinstance(data.index, pd.DatetimeIndex):
            pdf = pl.from_pandas(data.reset_index())
            idx_name = data.index.name or "_time"
            if pdf.columns[0] == "index":
                pdf = pdf.rename({"index": idx_name})
            return pdf
        return pl.from_pandas(data)
    if isinstance(data, pd.Series):
        return pl.from_pandas(data.to_frame())
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if columns:
            return pl.DataFrame({name: data[:, i] for i, name in enumerate(columns)})
        return pl.DataFrame({f"col_{i}": data[:, i] for i in range(data.shape[1])})
    if isinstance(data, dict):
        return pl.DataFrame(data)
    raise TypeError(f"Cannot convert {type(data).__name__} to polars DataFrame.")


def to_pandas(data: object) -> "pd.DataFrame":
    """Convert ``data`` to a :class:`pandas.DataFrame`.

    Accepts:
    - :class:`pandas.DataFrame` → returned unchanged
    - :class:`polars.DataFrame` → converted via ``pl.to_pandas``
    - :class:`polars.Series` → converted via ``pl.Series.to_pandas()``
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pd.Series):
        return data.to_frame()

    import polars as pl

    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    if isinstance(data, pl.Series):
        return data.to_pandas().to_frame()
    if isinstance(data, np.ndarray):
        cols = [f"col_{i}" for i in range(data.shape[1])] if data.ndim > 1 else ["value"]
        return pd.DataFrame(data, columns=cols)
    raise TypeError(f"Cannot convert {type(data).__name__} to pandas DataFrame.")


def is_polars(data: object) -> bool:
    """Check if ``data`` is a polars DataFrame/Series."""
    try:
        import polars as pl

        return isinstance(data, (pl.DataFrame, pl.Series))
    except ImportError:
        return False


def is_pandas(data: object) -> bool:
    """Check if ``data`` is a pandas DataFrame/Series."""
    try:
        import pandas as pd

        return isinstance(data, (pd.DataFrame, pd.Series))
    except ImportError:
        return False


def to_numpy(data: object) -> np.ndarray:
    """Convert DataFrame/Series to numpy, regardless of backend."""
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "to_numpy"):
        return data.to_numpy()
    return np.asarray(data, dtype=float)


def with_columns(data: "pl.DataFrame", **kwargs: object) -> "pl.DataFrame":
    """Add or replace columns in a polars DataFrame.

    Equivalent to ``pandas``' ``df[col] = values`` pattern.
    """
    import polars as pl

    for name, value in kwargs.items():
        if isinstance(value, (list, np.ndarray)):
            data = data.with_columns(pl.Series(name, value).alias(name))
        elif isinstance(value, pl.Expr):
            data = data.with_columns(value.alias(name))
        elif isinstance(value, (int, float, str)):
            data = data.with_columns(pl.lit(value).alias(name))
        else:
            raise TypeError(f"Unsupported column type: {type(value).__name__}")
    return data


def has_datetime_column(df: "pl.DataFrame") -> bool:
    """Check if a polars DataFrame has any datetime or date column."""
    import polars as pl

    return any(
        s in (pl.Datetime, pl.Date, pl.Datetime("ms", "UTC"), pl.Datetime("us", "UTC"))
        for s in df.schema.values()
    )


def datetime_columns(df: "pl.DataFrame") -> list[str]:
    """Return names of columns with datetime/date dtype."""
    import polars as pl

    valid = (pl.Datetime, pl.Date, pl.Datetime("ms", "UTC"), pl.Datetime("us", "UTC"))
    return [c for c, t in df.schema.items() if t in valid]


def is_pandas_datetime_index(df) -> bool:
    """Check if df is a pandas DataFrame with a DatetimeIndex."""
    try:
        import pandas as pd

        return isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex)
    except ImportError:
        return False
