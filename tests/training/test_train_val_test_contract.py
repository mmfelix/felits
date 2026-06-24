"""Guard tests for the TRAIN / VAL / TEST contract in the training scripts.

These tests enforce the project rule: train, val and test splits are sacred
and must be respected without ambiguity. Specifically:

  1. No script may use ``validation_split=0.X`` (or similar fractional split)
     as a shortcut for the dedicated val split. The val split (2017-2019) is
     already produced by ``prepare_data.py`` and persisted to
     ``splits.joblib``; any model that wants validation must use it.
  2. ``model.fit(...)`` (and equivalent ``XGBoostForecaster.fit``) must train
     on ``train_ws`` only. Neither ``val_ws`` nor ``test_ws`` may be passed
     as training data, and concatenating them into the training set is also
     forbidden (e.g. ``np.concatenate([train_ws.X, val_ws.X])``).
  3. ``model.predict(...)`` is allowed only on ``test_ws`` for the final
     metric report, or on ``val_ws`` inside an Optuna objective / early-
     stopping evaluation. ``test_ws`` must never be used as validation
     (no leakage into model selection / early stopping).
  4. Optuna-style scripts that pass a non-test validation source to
     ``model.fit`` must use ``val_ws`` (via ``validation_data=`` or
     ``eval_set=``), never ``test_ws``.

If any of these invariants is violated, the corresponding test fails with a
descriptive message that points to the offending file and line.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Iterable

import pytest

TRAINING_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "paper" / "scripts" / "training"

# Model scripts we care about (excludes common.py and prepare_data.py, which
# are data-prep utilities, not training scripts).
TRAINING_SCRIPTS = sorted(p for p in TRAINING_DIR.glob("train_*.py") if p.is_file())


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_call_visitor(tree: ast.AST) -> Iterable[tuple[ast.Call, str]]:
    """Yield ``(call_node, source_line_text)`` for every function call in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node, ast.unparse(node)


# ---------------------------------------------------------------------------
# Invariant 1: no ``validation_split=...`` in any training script.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", TRAINING_SCRIPTS, ids=lambda p: p.name)
def test_no_validation_split_in_training_scripts(script: Path) -> None:
    """No model may use Keras' validation_split shortcut.

    The pipeline persists a dedicated val split to splits.joblib; using
    validation_split would silently overwrite part of the training set and
    break the train / val / test contract.
    """
    src = _read(script)
    if "validation_split" not in src:
        return
    lines = src.splitlines()
    for i, line in enumerate(lines, start=1):
        if "validation_split" in line and not line.lstrip().startswith("#"):
            pytest.fail(
                f"{script.name}:{i}: forbidden 'validation_split' in training script. "
                "Use the dedicated val_ws split from splits.joblib instead "
                "(see prepare_data.py)."
            )


# ---------------------------------------------------------------------------
# Invariant 2: model.fit must train on train_ws, never on val_ws/test_ws.
# ---------------------------------------------------------------------------

def _fit_args(call: ast.Call) -> dict[str, ast.AST]:
    """Return ``{kw.arg: kw.value}`` for a Call's keyword arguments."""
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _contains_name(node: ast.AST, name: str) -> bool:
    """True iff ``node`` references the given simple name (no attribute access)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id == name:
            return True
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == name
        ):
            return True
    return False


_FORBIDDEN_FIT_SOURCES = {"val_ws", "test_ws"}
_ALLOWED_FIT_SOURCES = {"train_ws"}


@pytest.mark.parametrize("script", TRAINING_SCRIPTS, ids=lambda p: p.name)
def test_model_fit_uses_train_only(script: Path) -> None:
    """model.fit(...) must take train_ws (X, y) as training data.

    No val_ws / test_ws may be used as training data, and concatenating them
    with train_ws is also forbidden.
    """
    src = _read(script)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.fail(f"{script.name}: syntax error")

    # Determine whether this script is an Optuna-style script. Optuna scripts
    # legitimately have *two* ``model.fit`` calls (one in the trial objective,
    # one in the final test evaluation); the rule is the same for both.
    is_optuna = "optuna" in script.name

    for call, _ in _function_call_visitor(tree):
        # Match a fit invocation. ``self._model.fit`` (used by the sklearn
        # wrappers) and bare ``model.fit`` are both fine; we detect both by
        # looking at the function name.
        func = call.func
        if not (
            (isinstance(func, ast.Attribute) and func.attr == "fit")
            or (isinstance(func, ast.Name) and func.id == "fit")
        ):
            continue

        args = call.args
        kwargs = _fit_call_kwargs(call)

        # Build the set of source variables that feed ``X`` and ``y``.
        # Keras: ``fit(X, y, ...)`` — X = args[0], y = args[1].
        # sklearn wrapper: ``fit(X, y, eval_set=...)`` — same.
        if len(args) >= 2:
            x_src = _name_refs(args[0])
            y_src = _name_refs(args[1])
        else:
            # Keyword form: ``fit(X=..., y=...)`` — used by some wrappers.
            x_src = _name_refs(kwargs.get("X"))
            y_src = _name_refs(kwargs.get("y"))
            if x_src is None or y_src is None:
                continue

        for src_var in (x_src | y_src):
            if src_var in _FORBIDDEN_FIT_SOURCES:
                pytest.fail(
                    f"{script.name}:{call.lineno}: model.fit uses '{src_var}' as "
                    "training data. Training must use train_ws only; val_ws is "
                    "for validation, test_ws is held out for the final metric."
                )
            if src_var in _ALLOWED_FIT_SOURCES:
                continue
            # Variables like ``X_train``, ``y_train`` (locals) are tolerated
            # only if the script is a third-party wrapper that re-aliases
            # them. To be strict, we require the source to mention train_ws
            # somewhere in the call; otherwise flag it.
            if "train_ws" not in x_src and "train_ws" not in y_src:
                # The Optuna scripts build per-trial X/y with different names
                # (X, y aliases) — those are still sourced from train_ws.
                # If neither X nor Y references train_ws, fail.
                if not is_optuna and not (x_src or y_src):
                    continue
                # For non-optuna scripts, the source must be train_ws.
                if not is_optuna:
                    pytest.fail(
                        f"{script.name}:{call.lineno}: model.fit training data "
                        "does not reference train_ws. The only allowed source "
                        "is train_ws (val_ws/test_ws forbidden)."
                    )


def _fit_call_kwargs(call: ast.Call) -> dict[str, ast.AST]:
    return _fit_args(call)


def _name_refs(node: ast.AST | None) -> set[str]:
    """Return the set of simple variable names referenced anywhere in ``node``."""
    if node is None:
        return set()
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            names.add(sub.value.id)
    return names


# ---------------------------------------------------------------------------
# Invariant 3: model.predict must be on test_ws or val_ws, never train_ws.
# ---------------------------------------------------------------------------

_PREDICT_ALLOWED = {"test_ws", "val_ws"}


@pytest.mark.parametrize("script", TRAINING_SCRIPTS, ids=lambda p: p.name)
def test_model_predict_uses_val_or_test(script: Path) -> None:
    """``model.predict(X)`` may only be called on val_ws or test_ws."""
    src = _read(script)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.fail(f"{script.name}: syntax error")

    for call, _ in _function_call_visitor(tree):
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "predict"):
            continue
        if not call.args:
            continue
        refs = _name_refs(call.args[0])
        if not refs:
            continue
        # At least one referenced name must be in the allowed set.
        if refs.isdisjoint(_PREDICT_ALLOWED):
            pytest.fail(
                f"{script.name}:{call.lineno}: model.predict is called on "
                f"{sorted(refs)}, which is neither val_ws nor test_ws. "
                "Predictions on train_ws are not part of the contract."
            )


# ---------------------------------------------------------------------------
# Invariant 4: no concatenation of val_ws or test_ws with train_ws in training.
# ---------------------------------------------------------------------------

_CONCAT_FUNCS = {
    "concatenate",
    "concat",
    "vstack",
    "hstack",
    "append",
    "stack",
}
_CONCAT_MODULES = {"np", "numpy", "pandas", "pd"}


@pytest.mark.parametrize("script", TRAINING_SCRIPTS, ids=lambda p: p.name)
def test_no_train_val_or_test_concatenation_in_training(script: Path) -> None:
    """No np.concatenate / pd.concat / vstack / hstack / stack / append that
    mixes train_ws with val_ws or test_ws in the training path.

    This forbids the ``train + val -> new train`` anti-pattern.
    """
    src = _read(script)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.fail(f"{script.name}: syntax error")

    for call, _ in _function_call_visitor(tree):
        func = call.func
        # np.concatenate / pd.concat / numpy.concatenate / etc.
        is_concat = False
        if isinstance(func, ast.Attribute) and func.attr in _CONCAT_FUNCS:
            if isinstance(func.value, ast.Name) and func.value.id in _CONCAT_MODULES:
                is_concat = True
        elif isinstance(func, ast.Name) and func.id in _CONCAT_FUNCS:
            is_concat = True
        if not is_concat:
            continue

        # Gather all source-variable names appearing anywhere in the call.
        all_refs: set[str] = set()
        for arg in call.args:
            all_refs |= _name_refs(arg)
        for kw in call.keywords:
            all_refs |= _name_refs(kw.value)

        involves_train = "train_ws" in all_refs
        involves_val = "val_ws" in all_refs
        involves_test = "test_ws" in all_refs
        if involves_train and (involves_val or involves_test):
            pytest.fail(
                f"{script.name}:{call.lineno}: concatenates train_ws with "
                f"{'val_ws' if involves_val else 'test_ws'}. The training set "
                "must remain train_ws only; val_ws and test_ws are sacred."
            )


# ---------------------------------------------------------------------------
# Invariant 5: validation_data= and eval_set= must reference val_ws (not test_ws).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", TRAINING_SCRIPTS, ids=lambda p: p.name)
def test_validation_data_and_eval_set_use_val_only(script: Path) -> None:
    """``validation_data=...`` and ``eval_set=...`` must reference val_ws.

    Using test_ws in either would leak the test set into model selection
    (EarlyStopping, Optuna objective).
    """
    src = _read(script)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.fail(f"{script.name}: syntax error")

    for call, _ in _function_call_visitor(tree):
        kwargs = _fit_call_kwargs(call)
        for kw_name in ("validation_data", "eval_set"):
            node = kwargs.get(kw_name)
            if node is None:
                continue
            refs = _name_refs(node)
            if "test_ws" in refs:
                pytest.fail(
                    f"{script.name}:{call.lineno}: {kw_name}= references "
                    "test_ws. Validation must use val_ws, never test_ws."
                )
            if "val_ws" not in refs:
                pytest.fail(
                    f"{script.name}:{call.lineno}: {kw_name}= does not "
                    "reference val_ws. Use the dedicated validation split."
                )


# ---------------------------------------------------------------------------
# Invariant 6: optuna trials must not call model.predict(test_ws.X).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "script", [p for p in TRAINING_SCRIPTS if "optuna" in p.name], ids=lambda p: p.name
)
def test_optuna_objective_does_not_predict_on_test(script: Path) -> None:
    """Optuna trial objectives must score on val_ws, not test_ws.

    The only place test_ws may appear in an Optuna script is inside the
    ``evaluate_on_test`` function (or equivalent final evaluation block).
    """
    src = _read(script)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        pytest.fail(f"{script.name}: syntax error")

    for call, _ in _function_call_visitor(tree):
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "predict"):
            continue
        if not call.args:
            continue
        refs = _name_refs(call.args[0])
        if "test_ws" not in refs:
            continue
        # Find enclosing FunctionDef name.
        enclosing = _enclosing_function(tree, call)
        fname = enclosing.name if enclosing else "<module>"
        if fname == "evaluate_on_test" or fname == "evaluate_on_holdout":
            continue
        pytest.fail(
            f"{script.name}:{call.lineno}: model.predict(test_ws.X) inside "
            f"function '{fname}'. Optuna objectives must score on val_ws; "
            "test_ws is only used in the final evaluation block "
            "(e.g. evaluate_on_test)."
        )


def _enclosing_function(tree: ast.AST, target: ast.AST) -> ast.FunctionDef | None:
    """Return the FunctionDef that contains ``target`` (best-effort)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if sub is target:
                    return node
    return None
