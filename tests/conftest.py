from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from fluxopt import Flow, Port

if TYPE_CHECKING:
    from collections.abc import Sequence


def ts(n: int) -> list[datetime]:
    """Create *n* hourly timesteps starting 2024-01-01.

    Args:
        n: Number of timesteps to generate.
    """
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(n)]


class Table:
    """A tidy result table, read the way an assertion asks: pick labels, then take the numbers.

    ``sel`` filters on labels and drops those columns; ``values`` lays what is
    left out as an array, one axis per remaining key column, each in the order
    its labels first appear (time and period in their own order).
    """

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(c for c in self.frame.columns if c != 'value')

    def labels(self, dim: str) -> list[Any]:
        column = self.frame.get_column(dim)
        return (
            column.unique().sort().to_list()
            if dim in ('time', 'period')
            else column.unique(maintain_order=True).to_list()
        )

    def sel(self, **labels: Any) -> Table:
        frame = self.frame
        for dim, label in labels.items():
            frame = frame.filter(pl.col(dim) == label).drop(dim)
            if frame.is_empty():
                raise KeyError(f'no row with {dim}={label!r}')
        return Table(frame)

    def rename(self, **names: str) -> Table:
        return Table(self.frame.rename(names))

    @property
    def values(self) -> np.ndarray:
        dims = self.dims
        if not dims:
            return self.frame.get_column('value').to_numpy().reshape(())
        axes: Sequence[list[Any]] = [self.labels(d) for d in dims]
        out = np.full(tuple(len(a) for a in axes), np.nan)
        index = [{label: i for i, label in enumerate(a)} for a in axes]
        for row in self.frame.iter_rows(named=True):
            out[tuple(index[k][row[d]] for k, d in enumerate(dims))] = row['value']
        return out

    def item(self) -> float:
        if self.frame.height != 1:
            raise ValueError(f'item() needs one row, got {self.frame.height} over {self.dims}')
        return float(self.frame.item(0, 'value'))

    def __float__(self) -> float:
        return self.item()

    def sum(self, *dims: str) -> Table:
        """The table summed over *dims*, or over every dim when none are named."""
        keep = [d for d in self.dims if d not in dims] if dims else []
        total = pl.col('value').sum()
        return Table(self.frame.group_by(keep, maintain_order=True).agg(total) if keep else self.frame.select(total))

    def equals(self, other: Table) -> bool:
        return self.dims == other.dims and np.array_equal(self.values, other.values, equal_nan=True)


def read(result: Any, name: str, kind: str = 'primal') -> Table:
    """A variable, or with ``kind='expression'`` a reported expression, at *result*.

    A system that declares no periods is solved on one period labelled 0;
    the reader drops that axis, as a reader of such a system would.
    """
    frame = result.evaluate(name) if kind == 'expression' else getattr(result, kind)(name)
    if 'period' in frame.columns and frame.get_column('period').unique().to_list() == [0]:
        frame = frame.drop('period')
    return Table(frame)


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
