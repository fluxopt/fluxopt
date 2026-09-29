"""Shared helpers for ported mathematical correctness tests.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.

The ``optimize`` fixture is parametrized so every test runs three times,
each verifying a different pipeline:

``optimize``
    Baseline correctness check.
``archive->reload->solve``
    Proves the archived spec and sources solve again as they were.
``optimize->save->reload->validate``
    Proves specsolve's saved answer reads back the same.
"""

from __future__ import annotations

from typing import Any

import pytest
import specsolve
from conftest import read, ts, waste  # noqa: F401 — re-exported for test imports

from fluxopt import optimize as fluxopt_optimize


@pytest.fixture(
    params=[
        'optimize',
        'archive->reload->solve',
        'optimize->save->reload->validate',
    ]
)
def optimize(request, tmp_path):
    """Callable fixture: each test runs 3 pipelines to verify IO roundtrip."""

    def _optimize(**kwargs: Any) -> Any:
        objective = kwargs.pop('objective', 'cost')
        if request.param == 'optimize':
            return fluxopt_optimize(**kwargs, objective=objective)
        if request.param == 'archive->reload->solve':
            fluxopt_optimize(**kwargs, objective=objective, archive=tmp_path / 'run.zip')
            back = specsolve.load_archive(tmp_path / 'run.zip')
            return specsolve.solve(back.spec, back.sources)
        # optimize->save->reload->validate
        result = fluxopt_optimize(**kwargs, objective=objective)
        return specsolve.load_result(result.save(tmp_path / 'result'))

    _optimize.pipeline = request.param  # type: ignore[attr-defined]
    return _optimize
