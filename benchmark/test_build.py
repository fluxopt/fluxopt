"""Build-pipeline benchmarks — one ``benchmark()`` suite for CodSpeed and benchmem.

Run from this pinned, standalone project (``cd benchmark``)::

    uv run pytest . --codspeed                        # CodSpeed
    uv run pytest . --benchmark-only --benchmark-memory  # local memory (benchmem)

See benchmark/README.md.
"""

from __future__ import annotations

import pytest
from systems import SCALES, SCENARIO_SCALE, SCENARIOS, build_model, make_model_data, multi_node


@pytest.fixture(params=list(SCENARIOS), ids=list(SCENARIOS))
def scenario(request: pytest.FixtureRequest) -> object:
    return SCENARIOS[request.param]


@pytest.fixture(params=list(SCALES), ids=list(SCALES))
def scale(request: pytest.FixtureRequest) -> dict:
    return SCALES[request.param]


def test_feature_model_data(benchmark: object, scenario: object) -> None:
    """Elements → ModelData, per feature archetype."""
    benchmark(make_model_data, scenario, **SCENARIO_SCALE)  # type: ignore[operator]


def test_feature_model(benchmark: object, scenario: object) -> None:
    """ModelData → linopy model, per feature archetype."""
    data = make_model_data(scenario, **SCENARIO_SCALE)  # type: ignore[arg-type]
    benchmark(build_model, data)  # type: ignore[operator]


def test_scaling_model_data(benchmark: object, scale: dict) -> None:
    """Elements → ModelData for multi_node across scales."""
    benchmark(make_model_data, multi_node, **scale)  # type: ignore[operator]


def test_scaling_model(benchmark: object, scale: dict) -> None:
    """ModelData → linopy model for multi_node across scales."""
    data = make_model_data(multi_node, **scale)
    benchmark(build_model, data)  # type: ignore[operator]
