"""Benchmarks over the realistic reference systems in ``reference.py``.

Like the archetypes in ``systems.py``, they live in this directory and are
built against whichever fluxopt is installed — comparing two branches is
running the suite on each (``uv run`` re-syncs the editable install after a
``git switch``; see README.md, which also covers ``benchmem sweep``).

A quarter year keeps a multi-round pytest-benchmark run reasonable; the
full-year numbers come from ``uv run python reference.py``.
"""

from __future__ import annotations

import pytest
import reference as fx_benchmark

QUARTER_YEAR = 2190


@pytest.fixture(params=list(fx_benchmark.SYSTEMS))
def reference_system(request: pytest.FixtureRequest) -> str:
    return request.param


def test_reference_build(benchmark: object, reference_system: str) -> None:
    """Full pipeline (Elements -> sources -> specsolve model) for one reference system."""
    row = benchmark(fx_benchmark.measure, reference_system, QUARTER_YEAR)  # type: ignore[operator]
    extra_info = getattr(benchmark, 'extra_info', None)
    if extra_info is not None and isinstance(row, dict):
        # Element-layer labels + measured model size; keys absent on older
        # fluxopt versions are skipped so cross-ref comparisons keep working.
        # The row's 'time' (length of the time axis) is recorded as
        # 'timesteps' so an `extra:` column never shadows the time metric.
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
            if key in row:
                extra_info[label] = row[key]
