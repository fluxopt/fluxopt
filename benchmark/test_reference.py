"""Benchmarks over the realistic reference systems in ``reference.py``.

Unlike the archetypes in ``systems.py``, these are full models with realistic
data. A quarter year keeps a multi-round pytest-benchmark run reasonable; the
full-year numbers come from ``python reference.py``.
"""

from __future__ import annotations

import pytest
import reference

QUARTER_YEAR = 2190


@pytest.fixture(params=list(reference.SYSTEMS))
def reference_system(request: pytest.FixtureRequest) -> str:
    return request.param


def test_reference_build(benchmark: object, reference_system: str) -> None:
    """Full pipeline (Elements -> sources -> specsolve model) for one reference system."""
    row = benchmark(reference.measure, reference_system, QUARTER_YEAR)  # type: ignore[operator]
    extra_info = getattr(benchmark, 'extra_info', None)
    if extra_info is not None and isinstance(row, dict):
        # Element-layer labels + measured model size. The row's 'time' (length
        # of the time axis) is recorded as 'timesteps' so an `extra:` column
        # never shadows the time metric.
        for key, label in (
            ('time', 'timesteps'),
            ('periods', 'periods'),
            ('components', 'components'),
            ('flows', 'flows'),
            ('effects', 'effects'),
            ('series', 'series'),
            ('variables', 'variables'),
            ('nonzeros', 'nonzeros'),
            ('constraints', 'constraints'),
        ):
            extra_info[label] = row[key]
