"""Benchmarks over the bundled realistic reference systems (``fluxopt.benchmark``).

Unlike the archetypes in ``systems.py`` (which live in this directory and work
against any fluxopt version), these come from the installed package — which
makes them sweepable across versions or git refs::

    uv run --project benchmark benchmem sweep fluxopt \
        git+https://github.com/fluxopt/fluxopt@main \
        git+https://github.com/fluxopt/fluxopt@my-branch \
        --suite benchmark/ --memory

Versions that predate ``fluxopt.benchmark`` skip this file (importorskip).
A quarter year keeps a multi-round pytest-benchmark run reasonable; the
full-year numbers come from ``python -m fluxopt.benchmark``.
"""

from __future__ import annotations

import pytest

fx_benchmark = pytest.importorskip('fluxopt.benchmark')

QUARTER_YEAR = 2190


@pytest.fixture(params=['district_heating', 'industry_park', 'green_city', 'energy_transition'])
def reference_system(request: pytest.FixtureRequest) -> str:
    return request.param


def test_reference_build(benchmark: object, reference_system: str) -> None:
    """Full pipeline (Elements -> ModelData -> linopy model) for one reference system."""
    benchmark(fx_benchmark.measure, reference_system, QUARTER_YEAR)  # type: ignore[operator]
