"""Example 03: Feature selection (Granger, MI, mRMR, LASSO, SHAP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from felits.feature_selection import (
    FeatureSelector,
    adaptive_lasso_selection,
    granger_feature_selection,
    lasso_selection,
    mrmr_selection,
    rf_importance_selection,
    shap_feature_selection,
)


def main() -> None:
    rng = np.random.default_rng(0)
    n, p = 600, 10
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 4] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    df["target"] = y

    # 1) Granger causality
    gsel = granger_feature_selection(df, target="target", max_lag=5, significance=0.05)
    print(f"Granger-selected features: {gsel}")

    # 2) Mutual information (mRMR)
    mrmr_sel = mrmr_selection(df.drop(columns="target"), y, k_features=5, mi_method="sklearn")
    print(f"mRMR-selected features: {mrmr_sel}")

    # 3) LASSO
    lasso_res = lasso_selection(df.drop(columns="target"), y, alpha="auto", cv=3)
    print(f"LASSO-selected features (alpha={lasso_res.alpha:.4f}): {lasso_res.selected_features}")

    # 4) Adaptive LASSO
    ada = adaptive_lasso_selection(df.drop(columns="target"), y, alpha=0.05)
    print(f"Adaptive-LASSO-selected: {ada.selected_features}")

    # 5) RF importance
    rf_sel, _ = rf_importance_selection(df.drop(columns="target"), y, threshold=0.1)
    print(f"RF-selected: {rf_sel}")

    # 6) SHAP closed-loop
    rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(df.drop(columns="target"), y)
    sh = shap_feature_selection(rf, df.drop(columns="target"), y=y, threshold=0.1, max_iters=2)
    print(f"SHAP-selected: {sh.selected_features} (history: {len(sh.history)} iters)")

    # 7) Composed pipeline
    pipeline = FeatureSelector(
        steps=[
            ("rf", {"threshold": 0.05}),
            ("lasso", {"alpha": 0.1}),
        ]
    )
    result = pipeline.run(df.drop(columns="target"), y)
    print(f"Pipeline intersection: {result.selected_features}")


if __name__ == "__main__":
    main()
