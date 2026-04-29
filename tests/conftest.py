"""Shared fixtures and helpers for the fluxopt test suite.

The ``optimize`` fixture is parametrized so every test using it runs three
times, each verifying a different pipeline:

``optimize``
    Baseline correctness check.
``save->reload->optimize``
    Proves the ModelData definition survives IO.
``optimize->save->reload->validate``
    Proves the solution data survives IO and contributions sum correctly.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytest

from fluxopt import Flow, FlowSystem, ModelData, Port
from fluxopt import optimize as fluxopt_optimize
from fluxopt.results import Result


@pytest.fixture(
    params=[
        'optimize',
        'save->reload->optimize',
        'optimize->save->reload->validate',
    ]
)
def optimize(request, tmp_path):
    """Callable fixture: each test runs 3 pipelines to verify IO roundtrip."""

    def _optimize(**kwargs: Any) -> Result:
        objective_effects = kwargs.pop('objective_effects', 'cost')
        if request.param == 'optimize':
            return fluxopt_optimize(**kwargs, objective_effects=objective_effects)
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
            data.to_netcdf(path, mode='w')
            loaded = ModelData.from_netcdf(path)
            model = FlowSystem(loaded)
            return model.optimize(objective_effects=objective_effects)
        # optimize->save->reload->validate
        result = fluxopt_optimize(**kwargs, objective_effects=objective_effects)
        path = tmp_path / 'result.nc'
        result.to_netcdf(path)
        loaded = Result.from_netcdf(path)
        _ = loaded.stats.effect_contributions  # validate contributions survive IO roundtrip
        return loaded

    _optimize.pipeline = request.param  # type: ignore[attr-defined]
    return _optimize


def ts(n: int) -> list[datetime]:
    """Create *n* hourly timesteps starting 2024-01-01.

    Args:
        n: Number of timesteps to generate.
    """
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(n)]


def waste(carrier: str) -> Port:
    """Free-disposal port that absorbs excess on *carrier* at zero cost.

    Args:
        carrier: Carrier id string.
    """
    return Port(f'_waste_{carrier}', exports=[Flow(carrier)])


def _block_lengths(on: np.ndarray, *, active: bool) -> list[tuple[int, int]]:
    """Return (start_index, length) for each contiguous block.

    Args:
        on: Binary array (values > 0.5 are "on").
        active: True to find on-blocks, False to find off-blocks.
    """
    binary = np.asarray(on) > 0.5
    if not active:
        binary = ~binary
    if len(binary) == 0:
        return []
    changes = np.diff(binary.astype(np.int8))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if binary[0]:
        starts = np.concatenate([[0], starts])
    if binary[-1]:
        ends = np.concatenate([ends, [len(binary)]])
    return list(zip(starts.tolist(), (ends - starts).tolist(), strict=True))


def _check_blocks(
    blocks: list[tuple[int, int]],
    on: np.ndarray,
    label: str,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
) -> None:
    for start, length in blocks:
        if min_length is not None:
            assert length >= min_length, f'{label}-block of {length} < min {min_length} at t={start}: {on}'
        if max_length is not None:
            assert length <= max_length, f'{label}-block of {length} > max {max_length} at t={start}: {on}'


def assert_on_blocks(
    on: np.ndarray,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
) -> None:
    """Assert every contiguous on-block has duration in [min_length, max_length].

    Args:
        on: Binary on/off array (values > 0.5 are "on").
        min_length: Minimum allowed block length (inclusive).
        max_length: Maximum allowed block length (inclusive).
    """
    _check_blocks(_block_lengths(on, active=True), on, 'on', min_length=min_length, max_length=max_length)


def assert_off_blocks(
    on: np.ndarray,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    skip_leading: bool = True,
) -> None:
    """Assert every contiguous off-block has duration in [min_length, max_length].

    Args:
        on: Binary on/off array (values <= 0.5 are "off").
        min_length: Minimum allowed block length (inclusive).
        max_length: Maximum allowed block length (inclusive).
        skip_leading: If True, ignore the first off-block (may be carry-over from prior).
    """
    blocks = _block_lengths(on, active=False)
    if skip_leading and blocks and blocks[0][0] == 0:
        blocks = blocks[1:]
    _check_blocks(blocks, on, 'off', min_length=min_length, max_length=max_length)
