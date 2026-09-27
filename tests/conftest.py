from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from fluxopt import Flow, ModelData, Port

if TYPE_CHECKING:
    import xarray as xr


def ts(n: int) -> list[datetime]:
    """Create *n* hourly timesteps starting 2024-01-01.

    Args:
        n: Number of timesteps to generate.
    """
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(n)]


def read(result: Any, name: str, kind: str = 'primal') -> xr.DataArray:
    """A variable, or with ``kind='expression'`` a reported expression, at *result*.

    A system that declares no periods is solved on one period labelled 0;
    the reader drops that axis, as a reader of such a system would.
    """
    arr = result.to_dataarray(name, kind)
    if 'period' in arr.dims and arr.sizes['period'] == 1 and arr.coords['period'].item() == 0:
        arr = arr.squeeze('period', drop=True)
    return arr


def solve_data(data: ModelData, objective: str | dict[str, float] = 'cost') -> Any:
    """Solve a `ModelData` built or reloaded by hand, as `FlowSystem.optimize` would."""
    import specsolve

    from fluxopt.math import build_sources, objective_weights, program

    tables, coords = build_sources(data, objective_weights(data, objective))
    return specsolve.solve(program(data.dims.time_dtype).expand('sos'), {**tables, **coords})


def waste(carrier: str) -> Port:
    """Free-disposal port that absorbs excess on *carrier* at zero cost.

    Args:
        carrier: Carrier id string.
    """
    return Port(id=f'_waste_{carrier}', exports=[Flow(carrier=carrier)])


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
