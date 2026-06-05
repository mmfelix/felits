"""Example 05: XAI explainability (SHAP, LIME, closed-loop meta-optimizer)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from felits.feature_selection import shap_feature_selection, shap_interaction_selection
from felits.feature_selection.xai import lime_explain_instance
from felits.xai import deep_shap_selector


def main() -> None:
    rng = np.random.default_rng(0)
    n, p = 200, 6
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 3] + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])

    rf = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=1).fit(df, y)

    # 1) Global SHAP-based feature ranking
    sh = shap_feature_selection(rf, df, y=y, threshold=0.1, max_iters=2)
    print(f"SHAP-selected features: {sh.selected_features}")
    print(f"SHAP importances: {sh.importances}")

    # 2) SHAP interaction pairs
    pairs = shap_interaction_selection(rf, df, top_k=5)
    print("Top-5 SHAP interactions:", pairs)

    # 3) LIME local explanation
    instance = df.iloc[0].to_numpy()
    exp = lime_explain_instance(rf, df, instance, num_features=5)
    print("LIME local explanation:", next(iter(exp.as_map().values()))[:3], "...")

    # 4) Closed-loop SHAP meta-optimizer
    def factory(cols):
        sub = df[cols]
        return RandomForestRegressor(n_estimators=50, random_state=0).fit(sub, y)

    val_X = df.copy()
    val_y = y + 0.05 * rng.standard_normal(n)
    result = deep_shap_selector(
        factory, df, y, val_X=val_X, val_y=val_y, max_iters=3, threshold=0.1
    )
    print(f"Deep SHAP closed-loop result: {result.selected_features}")
    print(f"History: {result.history}")


if __name__ == "__main__":
    main()
