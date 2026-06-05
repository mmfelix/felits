from __future__ import annotations

import optuna
import pytest

from felits.optimization import OptunaOptimizer


def test_single_objective_study() -> None:
    def obj(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5, 5)
        return (x - 2) ** 2

    opt = OptunaOptimizer(n_trials=20, directions=("minimize",), seed=0)
    study = opt.run(obj)
    assert study is not None
    summary = opt.summary()
    assert summary.n_trials == 20
    assert len(summary.best_trials) == 1


def test_multi_objective_study() -> None:
    def obj(trial: optuna.Trial):
        x = trial.suggest_float("x", -5, 5)
        return (x - 2) ** 2, (x + 1) ** 2

    opt = OptunaOptimizer(n_trials=20, directions=("minimize", "minimize"), seed=0)
    opt.run(obj)
    df = opt.to_dataframe()
    assert len(df) == 20
    assert "values" in df.columns


def test_summary_requires_run() -> None:
    opt = OptunaOptimizer()
    with pytest.raises(RuntimeError):
        opt.summary()
