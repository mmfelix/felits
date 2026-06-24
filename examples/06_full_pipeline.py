"""Example 06: Full STLF (Short-Term Load Forecasting) pipeline.

End-to-end pipeline:
    1. Load data
    2. Preprocess (Hampel + STL)
    3. Build features (cyclic, lag, rolling, shift)
    4. Select features (mRMR + LASSO)
    5. Train RandomForest and LSTM forecasters
    6. Apply closed-loop SHAP meta-optimizer
    7. Compare metrics
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from felits.data import make_synthetic_ts
from felits.feature_extraction import cyclical_encode, lag_features, shift_features
from felits.feature_selection import FeatureSelector
from felits.models import RandomForestForecaster, is_dl_available
from felits.preprocessing import HampelFilter, Metrics, TimeSeriesScaler
from felits.xai import deep_shap_selector


def main() -> None:
    # 1) Data
    df = make_synthetic_ts(n_samples=24 * 90, n_features=2, seed=0)

    # 2) Preprocess
    filt = HampelFilter(window_size=12)
    df["y_clean"] = filt.transform(df["y"].to_numpy())

    # 3) Features
    df = cyclical_encode(df, period=24)
    df = lag_features(df, columns=["y_clean"], lags=[1, 24, 168], drop_na=True)
    df = shift_features(df, columns=["x0", "x1"], shifts=[24])  # t+24 day-ahead exog

    target = "y_clean"
    feature_cols = [c for c in df.columns if c not in {"y", "y_clean"}]
    data = df.dropna(subset=[*feature_cols, target]).copy()
    X = data[feature_cols]
    y = data[target]

    # 4) Feature selection
    pipeline = FeatureSelector(
        steps=[
            ("lasso", {"alpha": "auto", "cv": 3}),
        ]
    )
    result = pipeline.run(X, y)
    print(f"Selected features: {result.selected_features}")
    Xs = X[result.selected_features]

    # 5) Train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.2, shuffle=False)

    # 6a) RandomForest forecaster
    rf = RandomForestForecaster(n_estimators=100, random_state=0)
    rf.fit(X_tr.to_numpy(), y_tr.to_numpy())
    pred_rf = rf.predict(X_te.to_numpy())
    metrics_rf = Metrics(y_te.to_numpy(), pred_rf).dict_metrics()
    print("RandomForest metrics:", {k: round(v, 4) for k, v in metrics_rf.items()})

    # 6b) Optional: RNN baseline
    if is_dl_available():
        import tensorflow as tf

        from felits.models import RNNBasedModel

        df2 = data[[target, *feature_cols]].dropna()
        # Split BEFORE fitting the scaler to avoid target leakage from test into
        # the feature/target scaling statistics.
        cut_rows = int(0.8 * len(df2))
        train_df = df2.iloc[:cut_rows]
        test_df = df2.iloc[cut_rows:]
        scaler = TimeSeriesScaler(scaling_type="minmax").fit(train_df, target)
        scaled_train = scaler.transform(train_df)
        scaled_test = scaler.transform(test_df)
        scaled = np.vstack([scaled_train, scaled_test])
        y_idx = list(df2.columns).index(target)
        h, p = 24, 12
        Xs3, ys3 = [], []
        for i in range(len(scaled) - h - p + 1):
            Xs3.append(scaled[i : i + h, :])
            ys3.append(scaled[i + h : i + h + p, y_idx])
        Xs3, ys3 = np.asarray(Xs3, dtype="float32"), np.asarray(ys3, dtype="float32")
        cut = int(0.8 * len(Xs3))
        rnn = RNNBasedModel(
            model_type="LSTM", timesteps=h, features=Xs3.shape[2], num_units=32, output_units=p
        )
        rnn.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        rnn.fit(Xs3[:cut], ys3[:cut], epochs=2, batch_size=32, verbose=0, shuffle=False)
        pred = rnn.predict(Xs3[cut:], verbose=0)
        pred_inv = scaler.inverse_transform_target(pred)
        y_inv = scaler.inverse_transform_target(ys3[cut:])
        m = Metrics(y_inv, pred_inv).dict_metrics()
        print("LSTM metrics:", {k: round(v, 4) for k, v in m.items()})

    # 7) Closed-loop SHAP meta-optimizer
    def factory(cols):
        sub = X_tr[cols]
        return RandomForestRegressor(n_estimators=50, random_state=0).fit(sub, y_tr)

    cl = deep_shap_selector(factory, X_tr, y_tr, val_X=X_te, val_y=y_te, max_iters=3, threshold=0.1)
    print(f"Closed-loop SHAP meta-optimizer: {cl.selected_features}")


if __name__ == "__main__":
    main()
