from __future__ import annotations

from datetime import datetime
from math import prod
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The dimensions a [`Variate`][fluxopt.Variate] can vary over, in the order a
#: tidy table's key columns are read.
VARIATE_DIMS = ('time', 'period')


class ProfileRef(BaseModel):
    """Reference to a time-series stored outside the model definition.

    A serializable stand-in for an inline ``Variate`` array: the profile lives
    in a table and is named here, so structural definitions round-trip to
    YAML/JSON without inlining 8760-point series. The table is supplied at
    solve time through ``profiles``, and [`resolve`][fluxopt.ProfileRef.resolve]
    reads the profile out of it.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    table: str
    """Id of the table holding the profile (a key into ``profiles``)."""
    column: str
    """Column of that table holding the values."""

    def resolve(self, profiles: Mapping[str, pl.DataFrame]) -> pl.DataFrame | pl.Series:
        """Look up the referenced profile in *profiles*.

        Args:
            profiles: Mapping from table id to a table. Its ``time`` and
                ``period`` columns, where present, are the keys; every other
                column is a profile.

        Returns:
            A tidy table of the keys and ``value`` when the table has key
            columns, else the column itself, matched to the time axis by
            length.

        Raises:
            KeyError: If *table* is absent from *profiles*, or *column* from the table.
        """
        if self.table not in profiles:
            raise KeyError(f'ProfileRef table {self.table!r} not in profiles {sorted(profiles)}')
        table = profiles[self.table]
        if self.column not in table.columns:
            raise KeyError(f'ProfileRef column {self.column!r} not in table {self.table!r}')
        keys = [c for c in VARIATE_DIMS if c in table.columns]
        if not keys:
            return table.get_column(self.column)
        return table.select(*keys, pl.col(self.column).alias('value'))


# -- User input types --------------------------------------------------
type Variate = float | int | list[float] | np.ndarray | pl.Series | pl.DataFrame | ProfileRef
"""Any input that varies over a subset of the model's variate dims (``time``,
optionally ``period``).

- Scalar: broadcast to all variate dims.
- 1-D (``list``, ``np.ndarray``, ``pl.Series``): matched to a dim by length;
  ``time`` wins a tie, and any other tie is refused.
- Tidy table (``pl.DataFrame``): one column per dim it varies over, named
  after it, and a ``value`` column. It holds each combination of labels
  exactly once, and every label is one the model has.

Per-field reach (which dims a particular field can vary over) is documented on
the field itself; the aligner enforces that user input only uses dims the
caller declared.
"""

type Timesteps = list[datetime] | pl.Series

# -- Internal types (after normalization) ------------------------------
type TimeIndex = pl.Series


def variate_out_of_range(
    value: Variate,
    *,
    low: float,
    high: float,
    low_open: bool = False,
) -> float | None:
    """The first value outside ``[low, high]``, or None if all of them are in.

    A [`Variate`][fluxopt.Variate] is a scalar, a series, a table, or a
    [`ProfileRef`][fluxopt.ProfileRef] that names numbers living somewhere
    else. The first three can be checked where they are written, which is what
    this is for; a ``ProfileRef`` cannot, because its values arrive when
    profiles are resolved — so it reads as in range here and is checked again
    once it is real (``docs/design/validation-layers.md``).

    Args:
        value: The declared value.
        low: Lower bound.
        high: Upper bound.
        low_open: Whether *low* itself is excluded.

    Returns:
        An offending value, or None.
    """
    if isinstance(value, ProfileRef):
        return None
    values = np.atleast_1d(_numbers(value))
    below = values <= low if low_open else values < low
    bad = values[below | (values > high)]
    return float(bad[0]) if bad.size else None


def _numbers(value: Variate) -> np.ndarray:
    """The numbers a Variate holds, in the order it holds them."""
    if isinstance(value, pl.DataFrame):
        if 'value' not in value.columns:
            raise ValueError(f"A table Variate needs a 'value' column, got columns {value.columns}.")
        return value.get_column('value').cast(pl.Float64).to_numpy()
    if isinstance(value, pl.Series):
        return value.cast(pl.Float64).to_numpy()
    return np.asarray(value, dtype=float)


def align(value: Variate, axes: Mapping[str, pl.Series]) -> np.ndarray:
    """A Variate laid out on *axes*, as an array of shape ``len(axis)`` per axis.

    See [`Variate`][fluxopt.Variate] for the accepted inputs. A scalar fills
    the array, a 1-D value fills the one axis its length matches, and a tidy
    table fills the axes it has columns for; the rest broadcast.

    Args:
        value: The declared value.
        axes: The target axes, in the order of the result's dimensions. They
            are also the reach: a table column naming any other dim is
            refused.

    Raises:
        ValueError: If *value* is an unresolved ``ProfileRef``, a 1-D value
            whose length matches no axis or several, or a table with a foreign
            column, a label the model does not have, a duplicate row, or a
            missing combination.
        TypeError: If *value* is not a Variate.
    """
    if isinstance(value, ProfileRef):
        raise ValueError(
            f'Unresolved ProfileRef {value!r}: resolve it via ProfileRef.resolve(profiles) before building the model.'
        )
    shape = tuple(len(axis) for axis in axes.values())
    if isinstance(value, bool) or not isinstance(value, (int, float, list, np.ndarray, pl.Series, pl.DataFrame)):
        raise TypeError(f'Unsupported Variate type: {type(value).__name__}')
    if isinstance(value, (int, float)):
        return np.full(shape, float(value))
    if isinstance(value, pl.DataFrame):
        return _from_table(value, axes, shape)
    arr = _numbers(value)
    if arr.ndim != 1:
        raise ValueError(
            f'An array Variate must be 1-D (got ndim={arr.ndim}); '
            f'pass a tidy pl.DataFrame with one column per dim and a value column instead.'
        )
    return _from_1d(arr, axes, shape)


def _from_1d(arr: np.ndarray, axes: Mapping[str, pl.Series], shape: tuple[int, ...]) -> np.ndarray:
    """Length-match a 1-D array to one axis and broadcast it over the rest.

    Tie-breaking: ``time`` wins over other dims when several match. A tidy
    table names its dim and overrides this.
    """
    names = list(axes)
    matches = [name for name in names if len(axes[name]) == len(arr)]
    if not matches:
        lengths = ', '.join(f'{name}({len(axis)})' for name, axis in axes.items())
        raise ValueError(f'Length {len(arr)} does not match any dimension: {lengths}')
    if len(matches) > 1 and 'time' not in matches:
        raise ValueError(
            f'Length {len(arr)} matches several dimensions: {matches}. '
            f'Pass a tidy pl.DataFrame that names its dim to disambiguate.'
        )
    dim = 'time' if 'time' in matches else matches[0]
    view = [1] * len(names)
    view[names.index(dim)] = len(arr)
    return np.broadcast_to(arr.reshape(view), shape).copy()


def _from_table(table: pl.DataFrame, axes: Mapping[str, pl.Series], shape: tuple[int, ...]) -> np.ndarray:
    """Place a tidy table's values on *axes*, broadcast over the dims it has no column for.

    Every row lands on one cell: a label the axis does not have, a row
    repeated, and a combination left out are each refused rather than read as
    missing data.
    """
    if 'value' not in table.columns:
        raise ValueError(f"A table Variate needs a 'value' column, got columns {table.columns}.")
    dims = [c for c in table.columns if c != 'value']
    foreign = [d for d in dims if d not in axes]
    if foreign:
        raise ValueError(
            f'Table has columns {foreign} that are not dimensions here; its key columns must be among {list(axes)}.'
        )
    positions = table
    for dim in dims:
        axis = axes[dim]
        index = pl.DataFrame({dim: axis, f'_{dim}': np.arange(len(axis))})
        try:
            positions = positions.with_columns(pl.col(dim).cast(axis.dtype))
        except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as exc:
            raise ValueError(
                f'Table column {dim!r} has dtype {table.schema[dim]}; the model labels it {axis.dtype}.'
            ) from exc
        positions = positions.join(index, on=dim, how='left')
        unknown = positions.filter(pl.col(f'_{dim}').is_null()).get_column(dim).unique().to_list()
        if unknown:
            raise ValueError(f'Table has {dim} labels the model does not: {unknown[:5]}')
    if dims and positions.select(dims).is_duplicated().any():
        raise ValueError(f'Table repeats a combination of {dims}; each one appears once.')
    expected = prod(len(axes[d]) for d in dims)
    if table.height != expected:
        raise ValueError(
            f'Table has {table.height} rows but {dims} have {expected} combinations; give a value for each one.'
        )
    names = list(axes)
    local = np.full(tuple(len(axes[d]) for d in dims), np.nan)
    local[tuple(positions.get_column(f'_{d}').to_numpy() for d in dims)] = (
        positions.get_column('value').cast(pl.Float64).to_numpy()
    )
    order = [d for d in names if d in dims]
    local = np.transpose(local, [dims.index(d) for d in order])
    view = [len(axes[n]) if n in dims else 1 for n in names]
    return np.broadcast_to(local.reshape(view), shape).copy()


def normalize_timesteps(timesteps: Timesteps) -> TimeIndex:
    """Normalize user-provided timesteps to a datetime column named ``time``.

    Args:
        timesteps: Datetime objects, or a ``pl.Series`` of datetimes.

    Raises:
        TypeError: If a timestep is not a timestamp. Numbered steps are
            written as ``pl.datetime_range(datetime(2020, 1, 1), ..., '1h', eager=True)``.
        ValueError: If timesteps are empty, not strictly monotonically
            increasing, or contain duplicates.
    """
    if len(timesteps) == 0:
        raise ValueError('Timesteps must not be empty')
    if isinstance(timesteps, pl.Series):
        if not timesteps.dtype.is_temporal() or timesteps.dtype == pl.Duration:
            raise TypeError(
                f'Timesteps must be timestamps, got {timesteps.dtype}. '
                "For numbered steps, pass pl.datetime_range(datetime(2020, 1, 1), end, '1h', eager=True)."
            )
        time = timesteps.cast(pl.Datetime('us')).rename('time')
    elif isinstance(timesteps, list) and all(isinstance(t, datetime) for t in timesteps):
        time = pl.Series('time', timesteps, dtype=pl.Datetime('us'))
    else:
        found = type(timesteps[0]).__name__
        raise TypeError(
            f'Timesteps must be timestamps, got {found}. '
            "For numbered steps, pass pl.datetime_range(datetime(2020, 1, 1), end, '1h', eager=True)."
        )
    if time.n_unique() != len(time):
        raise ValueError('Timesteps contain duplicates')
    if len(time) > 1 and not time.is_sorted():
        raise ValueError('Timesteps must be strictly monotonically increasing')
    return time


def compute_dt(timesteps: TimeIndex, dt: float | list[float] | None) -> np.ndarray:
    """Each timestep's duration in hours.

    When dt is None, it is the consecutive differences in hours, the first
    step taking the second's; a single timestep lasts 1.0.

    Args:
        timesteps: Time index.
        dt: Override timestep duration. Validated against timesteps length.
    """
    n = len(timesteps)
    if dt is not None:
        if isinstance(dt, (int, float)):
            return np.full(n, float(dt))
        if isinstance(dt, list):
            if len(dt) != n:
                raise ValueError(f'dt length {len(dt)} does not match timesteps length {n}')
            return np.array(dt, dtype=float)
        raise TypeError(f'Unsupported dt type: {type(dt)}')
    if n <= 1:
        return np.ones(n)
    diffs = np.diff(timesteps.to_numpy()) / np.timedelta64(1, 'h')
    return np.concatenate([diffs[:1], diffs]).astype(float)
