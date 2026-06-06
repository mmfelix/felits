"""Hyperparameter optimisation wrappers around Optuna.

The :class:`OptunaOptimizer` class centralises the boilerplate of
multi-objective Optuna studies. It mirrors the legacy
``model_optimization.py`` pattern: an objective function receives a
``trial`` and returns a tuple of objective values, and the study is
configured with the TPE sampler and a fixed seed for reproducibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import optuna
import pandas as pd

__all__ = ["OptunaOptimizer", "StudySummary"]


@dataclass
class StudySummary:
    """Compact summary of an Optuna study, suitable for logging or CSV export."""

    best_trials: list[dict] = field(default_factory=list)
    n_trials: int = 0
    directions: list[str] = field(default_factory=list)
    user_attrs: dict = field(default_factory=dict)


class OptunaOptimizer:
    """High-level wrapper around :class:`optuna.create_study`.

    Parameters
    ----------
    n_trials:
        Number of trials to run.
    directions:
        ``"minimize"`` or ``"maximize"``, one per objective. Multi-objective
        studies are supported (pass a list of length > 1).
    sampler:
        Optuna sampler. ``None`` uses ``TPESampler(seed=0)``.
    seed:
        Random seed for the default TPE sampler.
    """

    def __init__(
        self,
        n_trials: int = 100,
        directions: Sequence[str] = ("minimize",),
        sampler: optuna.samplers.BaseSampler | None = None,
        seed: int = 0,
    ):
        self.n_trials = n_trials
        self.directions = list(directions)
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed)
        self.sampler = sampler
        self.study_: optuna.Study | None = None

    def run(self, objective: Callable[[optuna.Trial], Sequence[float] | float]) -> optuna.Study:
        self.study_ = optuna.create_study(directions=self.directions, sampler=self.sampler)
        self.study_.optimize(objective, n_trials=self.n_trials)
        return self.study_

    def summary(self) -> StudySummary:
        if self.study_ is None:
            raise RuntimeError("Call `run` before `summary`.")
        out = StudySummary(
            n_trials=len(self.study_.trials),
            directions=[str(d) for d in self.study_.directions],
        )
        try:
            best = self.study_.best_trials
        except (AttributeError, ValueError):
            best = []
        for t in best:
            row = {"params": t.params, "values": list(t.values) if t.values else None}
            row.update(t.user_attrs)
            out.best_trials.append(row)
        return out

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd

        if self.study_ is None:
            raise RuntimeError("Call `run` before `to_dataframe`.")
        rows = []
        for t in self.study_.trials:
            row = {**t.params, "values": t.values, "state": str(t.state)}
            row.update(t.user_attrs)
            rows.append(row)
        return pd.DataFrame(rows)
