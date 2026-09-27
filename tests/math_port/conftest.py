"""Shared helpers for ported mathematical correctness tests.

Each test builds a tiny, analytically solvable optimization model and asserts
that the objective (or key solution variables) match a hand-calculated value.

The ``optimize`` fixture is parametrized so every test runs three times,
each verifying a different pipeline:

``optimize``
    Baseline correctness check.
``save->reload->optimize``
    Proves the ModelData definition survives IO.
``optimize->save->reload->validate``
    Proves specsolve's saved answer reads back the same.
"""

from __future__ import annotations

from typing import Any

import pytest
import specsolve
from conftest import read, ts, waste  # noqa: F401 — re-exported for test imports

from fluxopt import ModelData
from fluxopt import optimize as fluxopt_optimize
from fluxopt.math import build_sources, objective_weights, program


@pytest.fixture(
    params=[
        'optimize',
        'save->reload->optimize',
        'optimize->save->reload->validate',
    ]
)
def optimize(request, tmp_path):
    """Callable fixture: each test runs 3 pipelines to verify IO roundtrip."""

    def _optimize(**kwargs: Any) -> Any:
        objective = kwargs.pop('objective', 'cost')
        if request.param == 'optimize':
            return fluxopt_optimize(**kwargs, objective=objective)
        if request.param == 'save->reload->optimize':
            data = ModelData.build(
                kwargs['timesteps'],
                kwargs['carriers'],
                kwargs['effects'],
                kwargs['ports'],
                kwargs.get('converters'),
                kwargs.get('storages'),
                kwargs.get('dt'),
                periods=kwargs.get('periods'),
                period_weights=kwargs.get('period_weights'),
            )
            path = tmp_path / 'data.nc'
            data.save(path)
            loaded = ModelData.load(path)
            tables, coords = build_sources(loaded, objective_weights(loaded, objective))
            return specsolve.solve(program(loaded.dims.time_dtype).expand('sos'), {**tables, **coords})
        # optimize->save->reload->validate
        result = fluxopt_optimize(**kwargs, objective=objective)
        return specsolve.load_result(result.save(tmp_path / 'result'))

    _optimize.pipeline = request.param  # type: ignore[attr-defined]
    return _optimize
