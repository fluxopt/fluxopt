"""fluxopt's math program, and the tables a system binds to it.

:data:`PROGRAM` is the math, one YAML fragment per feature. :func:`build_sources`
builds, straight from the elements, every table the program declares: one
function per feature, each returning the tables it owns, keyed by the labels
the elements were written with — the timestamp (or step number), the period
year, and the qualified ids.

Sparsity is row absence: a parameter keeps its declared rank while its table
holds only the rows that exist. Every relation between entities is a table of
its own (`carrier_of`, `port_of`, …) rather than a matrix.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from fluxopt.types import as_dataarray, compute_dt, normalize_timesteps
from fluxopt.validation import validate_system

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Carrier, Effect, Investment, Sizing, Status, Storage, _BoundFlow
    from fluxopt.types import Timesteps

#: The directory holding fluxopt's math, one YAML fragment per feature.
#: Shipped as package data.
PROGRAM = Path(__file__).with_name('program')


def program() -> Any:
    """fluxopt's math, composed from its fragments, loaded and checked.

    A :class:`mathspec.Spec`. Each file under :data:`PROGRAM` states one
    feature and loads on its own; ``effects.yaml`` declares the two halves of
    the ledger as sums, and every feature adds its own term to them, so
    ``merge`` writes the ledger. The engine verbs come from specsolve.
    """
    from mathspec import merge

    fragments = {path.stem: path for path in sorted(PROGRAM.glob('*.yaml'))}
    return merge(fragments, description='fluxopt: flows, converters and storages, and what they cost.')


#: The polars dtypes a table's columns take, by column name. A value column
#: is a float unless the parameter is declared ``dtype: bool``.
_STR, _INT, _FLOAT, _BOOL = pl.String, pl.Int64, pl.Float64, pl.Boolean


class UnsupportedFeatureError(RuntimeError):
    """The system uses a feature the program does not express yet.

    Raised rather than silently dropping the feature: a missing constraint
    would still solve, just to the wrong answer.
    """


# --- the horizon ------------------------------------------------------------


@dataclass(frozen=True)
class _Horizon:
    """The time and period axes, and the values laid out on them."""

    #: The timestep labels, as the user gave them.
    time: pd.Index
    #: Each timestep's duration [h].
    dt: np.ndarray
    #: The period labels, or ``[0]`` for a system that declares none.
    periods: list[int]
    #: Each period's weight; None for a system that declares no periods.
    period_weight: np.ndarray | None

    @property
    def n_time(self) -> int:
        return len(self.time)

    @property
    def n_period(self) -> int:
        return len(self.periods)

    @property
    def has_periods(self) -> bool:
        return self.period_weight is not None

    @property
    def time_series(self) -> pl.Series:
        """The time labels as a polars column."""
        return pl.Series('time', self.time)

    def on_grid(self, value: Any) -> np.ndarray:
        """A value on the `(time, period)` grid, time-major and flattened."""
        coords: dict[str, Any] = {'time': self.time}
        if self.has_periods:
            coords['period'] = pd.Index(self.periods)
        da = as_dataarray(value, coords)
        arr = np.asarray(da.transpose(*coords).values, dtype=float).reshape(self.n_time, -1)
        return np.broadcast_to(arr, (self.n_time, self.n_period)).ravel()

    def on_time(self, value: Any) -> np.ndarray:
        """A value per timestep."""
        return np.broadcast_to(np.asarray(as_dataarray(value, {'time': self.time}).values, dtype=float), (self.n_time,))

    def per_period(self, value: Any, what: str) -> list[float]:
        """A value per period: a scalar broadcasts, a sequence names each period."""
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.size not in (1, self.n_period):
            raise ValueError(
                f'{what} has {arr.size} values but the system has {self.n_period} period(s). '
                'Give one value per period, or a single value for all of them.'
            )
        return [float(arr[0])] * self.n_period if arr.size == 1 else [float(v) for v in arr]

    def grid(self, blocks: int) -> dict[str, Any]:
        """The `(time, period)` key columns for *blocks* whole grids stacked."""
        time = np.repeat(np.asarray(self.time), self.n_period)
        period = np.tile(np.asarray(self.periods, dtype=np.int64), self.n_time)
        return {'time': np.tile(time, blocks), 'period': np.tile(period, blocks)}

    def time_dtype(self) -> Any:
        return self.time_series.dtype


def _horizon(
    timesteps: Timesteps, dt: float | list[float] | None, periods: Any, period_weights: list[float] | None
) -> _Horizon:
    time = normalize_timesteps(timesteps)
    durations = np.asarray(compute_dt(time, dt).values, dtype=float)
    if periods is None:
        return _Horizon(time=time, dt=durations, periods=[0], period_weight=None)
    idx = pd.Index(periods, name='period')
    if not np.issubdtype(idx.dtype, np.integer):  # pyrefly: ignore[bad-argument-type]
        raise TypeError(f'periods must be integer, got {idx.dtype}')
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError('periods must be monotonically increasing and unique')
    if period_weights is not None:
        if len(period_weights) != len(idx):
            raise ValueError(f'period_weights has {len(period_weights)} entries, expected {len(idx)}')
        w = np.asarray(period_weights, dtype=float)
    elif len(idx) < 2:
        raise ValueError('period_weights is required when only one period is given')
    else:
        gaps = np.diff(idx.to_numpy().astype(int))
        w = np.append(gaps, gaps[-1]).astype(float)
    if not np.all(np.isfinite(w)) or not np.all(w > 0):
        raise ValueError(f'period_weights must be positive and finite, got {w}')
    return _Horizon(time=time, dt=durations, periods=[int(p) for p in idx], period_weight=w)


def _frame(columns: dict[str, Any], schema: dict[str, Any]) -> pl.DataFrame:
    """A typed table, NaN read as a missing value."""
    frame = pl.DataFrame({k: columns.get(k, []) for k in schema}, schema=schema)
    return frame.with_columns(pl.col(c).fill_nan(None) for c, t in schema.items() if t == _FLOAT)


def _flags(dim: str, ids: list[str]) -> pl.DataFrame:
    """A boolean table marking *ids* true."""
    return _frame({dim: ids, 'value': [True] * len(ids)}, {dim: _STR, 'value': _BOOL})


def _column(frame: pl.DataFrame, keys: list[str], column: str, *, drop_zero: bool = False) -> pl.DataFrame:
    """One column of *frame* as a `(keys..., value)` table, its live rows only."""
    out = frame.select([*keys, pl.col(column).alias('value')]).drop_nulls('value')
    return out.filter(pl.col('value') != 0) if drop_zero else out


def _per_step(
    keys: dict[str, list[Any]], values: list[np.ndarray], horizon: _Horizon, schema: dict[str, Any]
) -> pl.DataFrame:
    """Blocks of one value per timestep, each keyed by one entry of *keys*, stacked into a table."""
    n = horizon.n_time
    columns: dict[str, Any] = {name: np.repeat(np.asarray(v), n) if v else [] for name, v in keys.items()}
    columns['time'] = np.tile(np.asarray(horizon.time), len(values))
    columns['value'] = np.concatenate(values) if values else np.array([], dtype=float)
    return _frame(columns, {**schema, 'time': horizon.time_dtype(), 'value': _FLOAT})


def _on_periods(frame: pl.DataFrame, horizon: _Horizon, axis: str = 'period') -> pl.DataFrame:
    """A table that is the same in every period, crossed onto each of them."""
    return frame.join(pl.DataFrame({axis: horizon.periods}, schema={axis: _INT}), how='cross')


# --- flows ------------------------------------------------------------------


@dataclass(frozen=True)
class _Flows:
    """What the other features read about the flows."""

    bound: list[_BoundFlow]
    ids: list[str]
    #: flow id -> its fixed size, for flows sized to a number
    fixed_size: dict[str, float]
    sizing: list[tuple[str, Sizing]]
    invest: list[tuple[str, Investment]]
    #: flows whose rate follows a profile, in declaration order
    profiled: list[str]
    #: `(flow, time, period, relative_rate_min, relative_rate_max, fixed, size)`, dense
    grid: pl.DataFrame

    @property
    def has_size(self) -> set[str]:
        return set(self.fixed_size) | {i for i, _ in self.sizing} | {i for i, _ in self.invest}


def _flows(bound: list[_BoundFlow], horizon: _Horizon) -> tuple[_Flows, dict[str, Any]]:
    """The flow tables that need nothing but the flows."""
    from fluxopt.elements import Investment, Sizing

    ids = [bf.id for bf in bound]
    fixed_size = {
        bf.id: float(bf.flow.size)
        for bf in bound
        if bf.flow.size is not None and not isinstance(bf.flow.size, Sizing | Investment)
    }
    sizing = [(bf.id, bf.flow.size) for bf in bound if isinstance(bf.flow.size, Sizing)]
    invest = [(bf.id, bf.flow.size) for bf in bound if isinstance(bf.flow.size, Investment)]
    profiled = [bf.id for bf in bound if bf.flow.fixed_relative_profile is not None]
    cells = horizon.n_time * horizon.n_period

    grid = _frame(
        {
            'flow': np.repeat(ids, cells),
            **horizon.grid(len(ids)),
            'relative_rate_min': np.concatenate([horizon.on_grid(bf.flow.relative_rate_min) for bf in bound])
            if bound
            else [],
            'relative_rate_max': np.concatenate([horizon.on_grid(bf.flow.relative_rate_max) for bf in bound])
            if bound
            else [],
            'fixed': np.concatenate(
                [
                    horizon.on_grid(bf.flow.fixed_relative_profile)
                    if bf.flow.fixed_relative_profile is not None
                    else np.full(cells, np.nan)
                    for bf in bound
                ]
            )
            if bound
            else [],
            'size': np.repeat([fixed_size.get(i, np.nan) for i in ids], cells),
        },
        {
            'flow': _STR,
            'time': horizon.time_dtype(),
            'period': _INT,
            'relative_rate_min': _FLOAT,
            'relative_rate_max': _FLOAT,
            'fixed': _FLOAT,
            'size': _FLOAT,
        },
    )
    flows = _Flows(bound, ids, fixed_size, sizing, invest, profiled, grid)

    dt = pl.DataFrame({'time': horizon.time_series, 'dt': horizon.dt})
    sources: dict[str, Any] = {}

    ramp_rows = [bf for bf in bound if bf.flow.ramp_up_per_hour is not None or bf.flow.ramp_down_per_hour is not None]
    for kind in ('up', 'down'):
        declared = [bf for bf in ramp_rows if getattr(bf.flow, f'ramp_{kind}_per_hour') is not None]
        ramps = _frame(
            {
                'flow': np.repeat([bf.id for bf in declared], cells),
                **horizon.grid(len(declared)),
                'value': np.concatenate([horizon.on_grid(getattr(bf.flow, f'ramp_{kind}_per_hour')) for bf in declared])
                if declared
                else [],
            },
            {'flow': _STR, 'time': horizon.time_dtype(), 'period': _INT, 'value': _FLOAT},
        )
        # A ramp limit is per hour, so the step's allowance is limit x dt, per
        # unit of size. A ramp of 0 is a row: the row says the flow has a ramp.
        sources[f'ramp_{kind}_coeff'] = ramps.join(dt, on='time').select(
            ['flow', 'time', 'period', (pl.col('value') * pl.col('dt')).alias('value')]
        )

    pairs = [(bf.id, effect, factor) for bf in bound for effect, factor in bf.flow.effects_per_flow_hour.items()]
    per_flow_hour = _frame(
        {
            'flow': np.repeat([f for f, _, _ in pairs], cells),
            'effect': np.repeat([e for _, e, _ in pairs], cells),
            **horizon.grid(len(pairs)),
            'value': np.concatenate([horizon.on_grid(v) for _, _, v in pairs]) if pairs else [],
        },
        {'flow': _STR, 'effect': _STR, 'time': horizon.time_dtype(), 'period': _INT, 'value': _FLOAT},
    )
    # dt stays: a per-flow-hour rate times a duration is the step's energy.
    sources['effects_per_flow_hour'] = (
        per_flow_hour.join(dt, on='time')
        .select(['flow', 'effect', 'time', 'period', (pl.col('value') * pl.col('dt')).alias('value')])
        .filter(pl.col('value') != 0)
    )

    aggregated = [
        bf
        for bf in bound
        if any(
            v is not None
            for v in (bf.flow.flow_hours_min, bf.flow.flow_hours_max, bf.flow.load_factor_min, bf.flow.load_factor_max)
        )
    ]
    total_duration = float(horizon.dt.sum())
    for name in ('flow_hours_min', 'flow_hours_max', 'load_factor_min', 'load_factor_max'):
        # A load factor bounds the mean rate as a share of the size, so it
        # travels as lambda x T and the program multiplies by `size`.
        scale = total_duration if name.startswith('load_factor') else 1.0
        rows = [(bf.id, getattr(bf.flow, name) * scale) for bf in aggregated if getattr(bf.flow, name) is not None]
        sources[name] = _on_periods(
            _frame({'flow': [r[0] for r in rows], 'value': [r[1] for r in rows]}, {'flow': _STR, 'value': _FLOAT}),
            horizon,
        )
    return flows, sources


# --- sizing and investment --------------------------------------------------


def _lump_effects(
    items: list[tuple[str, Any]], fields: dict[str, str], entity: str, horizon: _Horizon
) -> dict[str, pl.DataFrame]:
    """One `(entity, effect, period, value)` table per coefficient field, declared pairs only."""
    out = {}
    for name, field in fields.items():
        rows = [
            (item_id, effect, period, value)
            for item_id, item in items
            for effect, raw in getattr(item, field).items()
            for period, value in zip(
                horizon.periods, horizon.per_period(raw, f'{item_id!r} {field}[{effect!r}]'), strict=True
            )
            if value != 0
        ]
        out[name] = pl.DataFrame(
            rows, schema={entity: _STR, 'effect': _STR, 'period': _INT, 'value': _FLOAT}, orient='row'
        )
    return out


def _sizing(flows: _Flows, horizon: _Horizon) -> dict[str, Any]:
    items = flows.sizing + flows.invest
    ids = [i for i, _ in items]
    invest_ids = [i for i, _ in flows.invest]
    sources: dict[str, Any] = {
        'has_sizing': _flags('flow', [i for i, _ in flows.sizing]),
        'has_invest': _flags('flow', invest_ids),
        # A flow's size is a `Sizing` or an `Investment`, never both, so the
        # two mechanisms share these without colliding.
        'mandatory': _flags('flow', [i for i, s in items if s.mandatory]),
        'size_min': _frame(
            {'flow': ids, 'value': [float(s.size_min) for _, s in items]}, {'flow': _STR, 'value': _FLOAT}
        ),
        'size_max': _frame(
            {'flow': ids, 'value': [float(s.size_max) for _, s in items]}, {'flow': _STR, 'value': _FLOAT}
        ),
        **_lump_effects(
            flows.sizing, {'effects_per_size': 'effects_per_size', 'effects_fixed': 'effects_fixed'}, 'flow', horizon
        ),
        **_lump_effects(
            flows.invest,
            {
                'effects_per_size_recurring': 'effects_per_size_recurring',
                'effects_fixed_recurring': 'effects_fixed_recurring',
            },
            'flow',
            horizon,
        ),
    }
    # A one-time cost belongs to the period the build happened in: the diagonal.
    for name, table in _lump_effects(
        flows.invest,
        {'effects_per_size_at_build': 'effects_per_size_at_build', 'effects_fixed_at_build': 'effects_fixed_at_build'},
        'flow',
        horizon,
    ).items():
        sources[name] = table.with_columns(pl.col('period').alias('build_period'))

    prior = [float(inv.prior_size) for _, inv in flows.invest]
    sources['prior_capacity'] = _frame({'flow': invest_ids, 'value': prior}, {'flow': _STR, 'value': _FLOAT})
    sources['has_prior_capacity'] = _flags('flow', [f for f, p in zip(invest_ids, prior, strict=True) if p > 0])
    window: list[tuple[str, Any, Any, float]] = []
    active: list[tuple[str, Any, float]] = []
    for (fid, inv), prior_size in zip(flows.invest, prior, strict=True):
        life = inv.lifetime
        for p_idx, period in enumerate(horizon.periods):
            for b_idx, build in enumerate(horizon.periods):
                if (b_idx <= p_idx) if life is None else (b_idx <= p_idx < b_idx + life):
                    window.append((fid, period, build, 1.0))
            alive = prior_size > 0 and (life is None or p_idx < life)
            active.append((fid, period, 1.0 if alive else 0.0))
    sources['lifetime_window'] = pl.DataFrame(
        window, schema={'flow': _STR, 'period': _INT, 'build_period': _INT, 'value': _FLOAT}, orient='row'
    )
    sources['prior_capacity_active'] = pl.DataFrame(
        active, schema={'flow': _STR, 'period': _INT, 'value': _FLOAT}, orient='row'
    )
    return sources


def _size_bound(flows: _Flows) -> pl.DataFrame:
    """Each flow's largest possible size, the big-M for a binary that releases a rate.

    Zero for a flow with no fixed size and no `Sizing`, which is what a flow
    that cannot carry anything is worth as a big-M.
    """
    upper = dict.fromkeys(flows.ids, 0.0)
    upper.update({i: float(s.size_max) for i, s in flows.sizing})
    upper.update(flows.fixed_size)
    return _frame({'flow': list(upper), 'value': list(upper.values())}, {'flow': _STR, 'value': _FLOAT})


# --- status -----------------------------------------------------------------


def _previous_duration(prior: list[float], state: int, dt: float) -> float:
    """How long *state* had held at the end of *prior*."""
    count = 0
    for v in reversed(prior):
        if (v > 0) == (state == 1):
            count += 1
        else:
            break
    return dt * count


def _status(
    entities: list[tuple[str, Status]],
    prior_rates: dict[str, list[float]],
    horizon: _Horizon,
) -> dict[str, Any]:
    """The tables of every on/off decision, a flow's or a component's, on one `status_entity` axis."""
    ids = [i for i, _ in entities]
    states = (('on', 'uptime'), ('off', 'downtime'))
    dt0 = float(horizon.dt[0])
    horizon_length = float(horizon.dt.sum())

    def per_state(rows: list[tuple[str, str, float | None]]) -> pl.DataFrame:
        return pl.DataFrame(
            rows, schema={'status_entity': _STR, 'state': _STR, 'value': _FLOAT}, orient='row'
        ).drop_nulls('value')

    minimum = per_state([(i, s, getattr(z, f'{k}_min')) for i, z in entities for s, k in states])
    maximum = per_state([(i, s, getattr(z, f'{k}_max')) for i, z in entities for s, k in states])
    ran = [i for i in ids if prior_rates.get(i) is not None]
    previous = per_state(
        [(i, s, _previous_duration(prior_rates[i], 1 if s == 'on' else 0, dt0)) for i in ran for s, _ in states]
    )
    # Big-M covers the whole horizon plus whatever ran before it.
    big_m = (
        pl.DataFrame({'status_entity': ids}, schema={'status_entity': _STR})
        .join(pl.DataFrame({'state': [s for s, _ in states]}), how='cross')
        .join(previous, on=['status_entity', 'state'], how='left')
        .select(['status_entity', 'state', (pl.col('value').fill_null(0.0) + horizon_length).alias('value')])
    )
    declared = pl.concat(
        [minimum.select(['status_entity', 'state']), maximum.select(['status_entity', 'state'])]
    ).unique(maintain_order=True)
    sources: dict[str, Any] = {
        'has_duration': declared.with_columns(pl.lit(True).alias('value')),
        'duration_min': minimum,
        'duration_big_m': big_m,
        # The counter's ceiling: the declared maximum, else the big-M.
        'duration_upper': big_m.join(maximum, on=['status_entity', 'state'], how='left', suffix='_max')
        .join(declared, on=['status_entity', 'state'])
        .select(['status_entity', 'state', pl.col('value_max').fill_null(pl.col('value')).alias('value')]),
        'previous_duration': previous,
        'initial_status': _frame(
            {'status_entity': ran, 'value': [1.0 if prior_rates[i][-1] > 0 else 0.0 for i in ran]},
            {'status_entity': _STR, 'value': _FLOAT},
        ),
        # A prior stay shorter than the minimum forces continuation.
        'forced_at_start': previous.join(minimum, on=['status_entity', 'state'], suffix='_min')
        .filter((pl.col('value') > 0) & (pl.col('value') < pl.col('value_min')))
        .select(['status_entity', 'state', pl.lit(True).alias('value')]),
    }
    cells = horizon.n_time * horizon.n_period
    for name, field, per_hour in (
        ('effects_per_running_hour', 'effects_per_running_hour', True),
        ('effects_per_startup', 'effects_per_startup', False),
    ):
        pairs = [(i, e, v) for i, z in entities for e, v in getattr(z, field).items()]
        table = _frame(
            {
                'status_entity': np.repeat([i for i, _, _ in pairs], cells),
                'effect': np.repeat([e for _, e, _ in pairs], cells),
                **horizon.grid(len(pairs)),
                'value': np.concatenate([horizon.on_grid(v) for _, _, v in pairs]) if pairs else [],
            },
            {'status_entity': _STR, 'effect': _STR, 'time': horizon.time_dtype(), 'period': _INT, 'value': _FLOAT},
        )
        if per_hour:
            # `dt` turns a per-running-hour rate into the step's cost; a startup
            # happens once at the step, so it is not scaled.
            dt = pl.DataFrame({'time': horizon.time_series, 'dt': horizon.dt})
            table = table.join(dt, on='time').with_columns(pl.col('value') * pl.col('dt')).drop('dt')
        sources[name] = table.filter(pl.col('value') != 0)
    return sources


# --- converters -------------------------------------------------------------


def _converters(converters: list[Converter], horizon: _Horizon) -> tuple[dict[str, Any], dict[str, str], int]:
    """Linear conversion: the non-zero coefficient of each flow in each equation.

    Returns the tables, the flow -> converter map, and the most equations any
    converter states.
    """
    linear = [c for c in converters if c.conversion is None]
    keys: dict[str, list[Any]] = {'flow': [], 'eq_idx': []}
    values: list[np.ndarray] = []
    converter_of: dict[str, str] = {}
    for conv in linear:
        for fid, flow, _sign in conv._qualified_flows():
            for eq_i, equation in enumerate(conv.conversion_factors):
                if flow.short_id in equation:
                    converter_of[fid] = conv.id
                    keys['flow'].append(fid)
                    keys['eq_idx'].append(eq_i)
                    values.append(horizon.on_time(equation[flow.short_id]))
    factor = _per_step(keys, values, horizon, {'flow': _STR, 'eq_idx': _INT})
    sources = {
        'conversion_factor': factor.select(['flow', 'eq_idx', 'time', 'value']).filter(pl.col('value') != 0),
        'conversion_active': pl.DataFrame(
            [(c.id, i, True) for c in linear for i in range(len(c.conversion_factors))],
            schema={'converter': _STR, 'eq_idx': _INT, 'value': _BOOL},
            orient='row',
        ),
    }
    width = max((len(c.conversion_factors) for c in linear), default=0)
    return sources, converter_of, width


def _piecewise(converters: list[Converter], horizon: _Horizon) -> tuple[dict[str, Any], dict[str, str], int]:
    """Piecewise curves: a link is a row on `flow`, its breakpoints the rows on `bp`.

    Returns the tables, the flow -> converter map, and the widest curve.
    """
    curves = [c for c in converters if c.conversion is not None]
    identity: list[tuple[str, str, Any]] = []
    present: list[tuple[str, int, bool]] = []
    value_keys: dict[str, list[Any]] = {'flow': [], 'bp': []}
    value_blocks: list[np.ndarray] = []
    avail_keys: dict[str, list[Any]] = {'converter': []}
    avail_blocks: list[np.ndarray] = []
    for conv in curves:
        curve = conv.conversion
        assert curve is not None
        qualified = {bf.flow.short_id: bf.id for bf in conv._qualified_flows()}
        links = list(curve._iter_normalized())
        for short, points, bound in links:
            identity.append((qualified[short], conv.id, bound))
            for bp, point in enumerate(points):
                value_keys['flow'].append(qualified[short])
                value_keys['bp'].append(bp)
                value_blocks.append(horizon.on_time(point))
        present.extend((conv.id, bp, True) for bp in range(len(links[0][1])))
        # Availability scales the envelope of the reference link, a curve's first.
        widest = np.max([horizon.on_time(p) for p in links[0][1]], axis=0)
        availability = horizon.on_time(curve.availability)
        avail_keys['converter'].append(conv.id)
        avail_blocks.append(availability * widest)
        at_zero = [
            np.max([np.abs(horizon.on_time(pts[bp])) for _, pts, _ in links], axis=0) <= 1e-9
            for bp in range(len(links[0][1]))
        ]
        if curve.status is not None and any(z.any() for z in at_zero):
            warnings.warn(
                f'PiecewiseConversion on converter {conv.id!r} has Status, '
                'but the curve includes a (0, ..., 0) breakpoint. The '
                'optimizer can sit at zero with status=on, decoupling the '
                'binary from the actual operating state — Status features '
                'will not behave as expected. If you want Status to work '
                'as expected, drop the zero breakpoint so the only way to '
                'produce zero is status=off.',
                UserWarning,
                stacklevel=5,
            )
    links_frame = pl.DataFrame(identity, schema={'flow': _STR, 'converter': _STR, 'value': _STR}, orient='row')
    first = links_frame.group_by('converter', maintain_order=True).first()
    sources = {
        'pw_bp_value': _per_step(value_keys, value_blocks, horizon, {'flow': _STR, 'bp': _INT})
        .select(['flow', 'bp', 'time', 'value'])
        .filter(pl.col('value') != 0),
        # Curves of different width share one `bp` axis, so the mask is what
        # stops a weight existing past the end of a narrower one.
        'pw_bp_present': pl.DataFrame(present, schema={'converter': _STR, 'bp': _INT, 'value': _BOOL}, orient='row'),
        'curve_of': links_frame.select(['flow', 'converter']),
        'link_sense': links_frame.select(['flow', 'value']),
        'pw_ref': first.select(['flow', pl.lit(1.0).alias('value')]),
        'pw_avail_bound': _on_periods(
            _per_step(avail_keys, avail_blocks, horizon, {'converter': _STR}).select(['converter', 'time', 'value']),
            horizon,
        ),
        # A gated curve's Status is keyed by the converter's own id.
        'pw_status_of': _frame(
            {'converter': [c.id for c in curves if c.conversion.status is not None]}  # type: ignore[union-attr]
            | {'status_entity': [c.id for c in curves if c.conversion.status is not None]},  # type: ignore[union-attr]
            {'converter': _STR, 'status_entity': _STR},
        ),
    }
    width = max((len(next(iter(c.conversion._iter_normalized()))[1]) for c in curves), default=0)  # type: ignore[union-attr]
    return sources, {fid: conv for fid, conv, _ in identity}, width


# --- storages ---------------------------------------------------------------


def _storages(storages: list[Storage], horizon: _Horizon) -> dict[str, Any]:
    from fluxopt.elements import Investment, Sizing

    ids = [s.id for s in storages]
    fixed = {
        s.id: float(s.capacity)
        for s in storages
        if s.capacity is not None and not isinstance(s.capacity, Sizing | Investment)
    }
    sized = [(s.id, s.capacity) for s in storages if isinstance(s.capacity, Sizing)]
    sized_ids = [i for i, _ in sized]

    def per_step(pick: Any) -> pl.DataFrame:
        blocks = [horizon.on_time(pick(s)) for s in storages]
        return _per_step({'storage': ids}, blocks, horizon, {'storage': _STR}).select(['storage', 'time', 'value'])

    dt = pl.DataFrame({'time': horizon.time_series, 'dt': horizon.dt})
    loss = per_step(lambda s: s.relative_loss_per_hour).join(dt, on='time')
    eta_c = per_step(lambda s: s.eta_charge).join(dt, on='time')
    eta_d = per_step(lambda s: s.eta_discharge).join(dt, on='time')
    rel_min = per_step(lambda s: s.relative_level_min)
    rel_max = per_step(lambda s: s.relative_level_max)
    capacity = pl.DataFrame(
        {'storage': list(fixed), 'capacity': list(fixed.values())}, schema={'storage': _STR, 'capacity': _FLOAT}
    )

    def absolute(rel: pl.DataFrame, missing: float) -> pl.DataFrame:
        # An absent capacity is a decision: 0 and infinity are what the level
        # bounds read, while the relative pair does the bounding in rows.
        return rel.join(capacity, on='storage', how='left').select(
            ['storage', 'time', (pl.col('value') * pl.col('capacity')).fill_null(missing).alias('value')]
        )

    def sized_only(rel: pl.DataFrame) -> pl.DataFrame:
        return rel.filter(pl.col('storage').is_in(pl.Series(sized_ids, dtype=_STR).implode()) & (pl.col('value') != 0))

    levels = {s.id: s for s in storages}
    return {
        'port_of': pl.DataFrame(
            [
                (f, s.id, side)
                for s in storages
                for side, f in (('charge', s._charging_id), ('discharge', s._discharging_id))
            ],
            schema={'flow': _STR, 'storage': _STR, 'side': _STR},
            orient='row',
        ),
        'retention': loss.select(['storage', 'time', ((1 - pl.col('value')) ** pl.col('dt')).alias('value')]),
        'storage_coeff': pl.concat(
            [
                eta_c.select(
                    ['storage', pl.lit('charge').alias('side'), 'time', (pl.col('value') * pl.col('dt')).alias('value')]
                ),
                eta_d.select(
                    [
                        'storage',
                        pl.lit('discharge').alias('side'),
                        'time',
                        (-pl.col('dt') / pl.col('value')).alias('value'),
                    ]
                ),
            ]
        ),
        'level_min': absolute(rel_min, 0.0),
        'level_max': absolute(rel_max, np.inf),
        'given_capacity': capacity.rename({'capacity': 'value'}),
        'has_capacity_sizing': _flags('storage', sized_ids),
        'capacity_mandatory': _flags('storage', [i for i, z in sized if z.mandatory]),
        'capacity_min': _frame(
            {'storage': sized_ids, 'value': [float(z.size_min) for _, z in sized]}, {'storage': _STR, 'value': _FLOAT}
        ),
        'capacity_max': _frame(
            {'storage': sized_ids, 'value': [float(z.size_max) for _, z in sized]}, {'storage': _STR, 'value': _FLOAT}
        ),
        'relative_level_min': sized_only(rel_min),
        'relative_level_max': sized_only(rel_max),
        **_lump_effects(
            sized,
            {'effects_per_capacity': 'effects_per_size', 'effects_fixed_capacity': 'effects_fixed'},
            'storage',
            horizon,
        ),
        'is_cyclic': _flags('storage', [s.id for s in storages if s.cyclic]),
        'prevent_simultaneous': _flags('storage', [s.id for s in storages if s.prevent_simultaneous]),
        # Dense: the prior level sits on the constant side of the first
        # step's balance, where an absent row is a binding zero.
        'prior_level': _on_periods(
            _frame(
                {'storage': ids, 'value': [float(levels[i].prior_level or 0.0) for i in ids]},
                {'storage': _STR, 'value': _FLOAT},
            ),
            horizon,
        ),
        **{
            key: _frame(
                {
                    'storage': [s.id for s in storages if getattr(s, key) is not None],
                    'value': [getattr(s, key) for s in storages if getattr(s, key) is not None],
                },
                {'storage': _STR, 'value': _FLOAT},
            )
            for key in ('final_level_min', 'final_level_max')
        },
    }


# --- effects ----------------------------------------------------------------


def objective_weights(effect_ids: list[str], objective: str | dict[str, float]) -> dict[str, float]:
    """The weights the objective is minimised with: *objective*, and the penalty.

    The built-in penalty effect is added at 1.0 unless the caller named it,
    which is what makes a soft constraint cost something.
    """
    from fluxopt.elements import PENALTY_EFFECT_ID

    weights = {objective: 1.0} if isinstance(objective, str) else {k: float(v) for k, v in objective.items()}
    if PENALTY_EFFECT_ID not in weights and PENALTY_EFFECT_ID in set(effect_ids):
        weights[PENALTY_EFFECT_ID] = 1.0
    return weights


def _cross_effect_factor(factor: Any, horizon: _Horizon, effect: str, source: str) -> list[float]:
    """A `contribution_from` factor per period; a factor over time is refused with its rewrite."""
    what = f'Effect {effect!r} contribution_from {source!r}'
    over_time = isinstance(factor, xr.DataArray | pd.Series | pd.DataFrame) and 'time' in (
        factor.dims if isinstance(factor, xr.DataArray) else [factor.index.name, *getattr(factor, 'columns', [])]
    )
    if over_time or np.size(factor) not in (1, horizon.n_period):
        raise ValueError(
            f'{what} varies over time; a cross-effect factor is a scalar or one value per period. '
            f'Charge a time-varying price on the flows instead: effects_per_flow_hour={{{effect!r}: price * factor}} '
            f'beside the {source!r} coefficient.'
        )
    return horizon.per_period(factor, what)


def _refuse_cycles(effect_ids: list[str], edges: list[tuple[str, str]]) -> None:
    """Refuse a self-reference or a cycle in `contribution_from`; it would make `I - C` singular."""
    graph: dict[str, list[str]] = {e: [] for e in effect_ids}
    for effect, source in edges:
        if effect == source:
            raise ValueError(f'Effect {effect!r} cannot reference itself in contribution_from')
        graph[effect].append(source)
    state: dict[str, int] = dict.fromkeys(graph, 0)
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        path.append(node)
        for nxt in graph[node]:
            if state[nxt] == 1:
                return [*path[path.index(nxt) :], nxt]
            if state[nxt] == 0 and (cycle := visit(nxt)) is not None:
                return cycle
        path.pop()
        state[node] = 2
        return None

    for node in graph:
        if state[node] == 0 and (cycle := visit(node)) is not None:
            raise ValueError(f'Circular contribution_from dependency: {" -> ".join(cycle)}')


def _effects(effects: list[Effect], objective: dict[str, float], horizon: _Horizon) -> dict[str, Any]:
    ids = [e.id for e in effects]
    factors = [
        (e.id, source, period, value)
        for e in effects
        for source, factor in e.contribution_from.items()
        for period, value in zip(horizon.periods, _cross_effect_factor(factor, horizon, e.id, source), strict=True)
    ]
    _refuse_cycles(
        ids, sorted({(e, s) for e, s, _, v in factors if v != 0}, key=lambda es: (ids.index(es[0]), ids.index(es[1])))
    )

    # (I - C)^-1 - I: what one unit charged to an effect adds to every other,
    # through every chain. Zero on the diagonal, since a cycle is refused.
    share_rows = []
    n = len(ids)
    for p_idx, period in enumerate(horizon.periods):
        c = np.zeros((n, n))
        for effect, source, p, value in factors:
            if p == period:
                c[ids.index(effect), ids.index(source)] = value
        if not c.any():
            continue
        chained = np.linalg.inv(np.eye(n) - c) - np.eye(n)
        share_rows.extend(
            (ids[i], ids[j], period, float(chained[i, j])) for i in range(n) for j in range(n) if chained[i, j] != 0
        )
        del p_idx

    def at_period(value: Any, what: str) -> list[float | None]:
        return [None] * horizon.n_period if value is None else list(horizon.per_period(value, what))

    weights: dict[tuple[str, Any], float] = {}
    for e in effects:
        own = at_period(e.period_weights, f'{e.id!r} period_weights')
        for p_idx, period in enumerate(horizon.periods):
            fallback = float(horizon.period_weight[p_idx]) if horizon.period_weight is not None else 1.0
            given = own[p_idx]
            weights[(e.id, period)] = given if given is not None else fallback
    periodic = [
        (e.id, period, lo, hi)
        for e in effects
        if e.periodic_min is not None or e.periodic_max is not None
        for period, lo, hi in zip(
            horizon.periods,
            at_period(e.periodic_min, f'{e.id!r} periodic_min'),
            at_period(e.periodic_max, f'{e.id!r} periodic_max'),
            strict=True,
        )
    ]
    key = {'effect': _STR, 'period': _INT, 'value': _FLOAT}
    return {
        'share': pl.DataFrame(
            share_rows, schema={'effect': _STR, 'source': _STR, 'period': _INT, 'value': _FLOAT}, orient='row'
        ),
        'same': pl.DataFrame({'source': ids, 'effect': ids}, schema={'source': _STR, 'effect': _STR}),
        # Objective weight x period weight, folded into one parameter.
        'objective_weight': pl.DataFrame(
            [(e, p, objective.get(e, 0.0) * w) for (e, p), w in weights.items() if objective.get(e, 0.0) * w != 0],
            schema=key,
            orient='row',
        ),
        'period_weight': pl.DataFrame([(e, p, w) for (e, p), w in weights.items()], schema=key, orient='row'),
        'periodic_min': pl.DataFrame(
            [(e, p, lo) for e, p, lo, _ in periodic if lo is not None], schema=key, orient='row'
        ),
        'periodic_max': pl.DataFrame(
            [(e, p, hi) for e, p, _, hi in periodic if hi is not None], schema=key, orient='row'
        ),
        **{
            name: _frame(
                {
                    'effect': [e.id for e in effects if getattr(e, name) is not None],
                    'value': [getattr(e, name) for e in effects if getattr(e, name) is not None],
                },
                {'effect': _STR, 'value': _FLOAT},
            )
            for name in ('total_min', 'total_max')
        },
    }


# --- the whole system ---------------------------------------------------------


def build_sources(
    *,
    timesteps: Timesteps,
    carriers: list[Carrier],
    effects: list[Effect],
    ports: list[Port],
    objective: str | dict[str, float],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | None = None,
    periods: Any = None,
    period_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Every table :data:`PROGRAM` is bound to, for a system of resolved elements.

    Raises:
        UnsupportedFeatureError: If the system uses a feature the program does
            not express yet, rather than dropping it silently.
        ValueError: If a value the element could not check is out of reach of
            the program too: a `contribution_from` cycle, or a status flow's
            zero floor.
    """
    from fluxopt.elements import PENALTY_EFFECT_ID, Effect, node_id

    converters = converters or []
    storages = storages or []
    if not any(e.id == PENALTY_EFFECT_ID for e in effects):
        effects = [*effects, Effect(id=PENALTY_EFFECT_ID)]
    validate_system(
        carriers=carriers,
        effects=effects,
        ports=ports,
        converters=converters,
        storages=storages,
        objective=objective,
    )
    horizon = _horizon(timesteps, dt, periods, period_weights)
    bound = [bf for comp in (*ports, *converters, *storages) for bf in comp._qualified_flows()]
    flows, sources = _flows(bound, horizon)
    ids = flows.ids

    if flows.invest and not horizon.has_periods:
        raise UnsupportedFeatureError('investment requires multi-period optimization (periods must be specified)')

    # --- status: a flow's own decision, then each component's -----------------
    components = [(s.id, s.status, [s._charging_id, s._discharging_id]) for s in storages if s.status is not None]
    components += [
        (c.id, c.conversion.status, [bf.id for bf in c._qualified_flows()])
        for c in converters
        if c.conversion is not None and c.conversion.status is not None
    ]
    own = [(bf.id, bf.flow.status) for bf in bound if bf.flow.status is not None]
    entities = own + [(cid, z) for cid, z, _ in components]
    own_ids = [i for i, _ in own]
    prior_rates = {bf.id: bf.flow.prior_rates for bf in bound if bf.flow.prior_rates is not None}
    if prior_rates and not np.allclose(horizon.dt, horizon.dt[0]):
        warnings.warn(
            f'prior_rates with non-uniform dt: pre-horizon status durations assume the first '
            f'timestep duration ({float(horizon.dt[0])} h) for every prior step. If your prior steps had '
            f'different durations, adjust prior_rates to compensate.',
            UserWarning,
            stacklevel=3,
        )
    floor = flows.grid.filter(
        pl.col('flow').is_in(pl.Series(own_ids, dtype=_STR).implode()) & (pl.col('relative_rate_min') <= 0)
    )
    if len(floor):
        degenerate = floor['flow'].unique(maintain_order=True).to_list()
        raise ValueError(
            f'Status flows must have rel_lb > 0 (else on/off is indistinguishable); violated on {degenerate}'
        )
    sizing_ids = {i for i, _ in flows.sizing}
    if both := sorted(sizing_ids & set(own_ids) & set(flows.profiled)):
        raise UnsupportedFeatureError(f'fixed profile with status+sizing has no formulation: {both}')
    sources |= _status(entities, prior_rates, horizon)

    # A piecewise curve gates its own flows through its convexity row, so a
    # flow only a curve's Status governs is not gated a second time.
    status_of = {f: f for f in own_ids}
    status_of |= {
        fid: cid
        for cid, _z, governed in components
        for fid in governed
        if not any(c.id == cid and c.conversion is not None for c in converters)
    }

    # --- rate bounds: one chain of overrides over the dense envelope -----------
    invest_ids = [i for i, _ in flows.invest]
    has_size, profiled = flows.has_size, set(flows.profiled)
    bounded = [f for f in ids if f in has_size and f not in profiled]

    def among(names: Any) -> pl.Expr:
        return pl.col('flow').is_in(pl.Series(sorted(names), dtype=_STR).implode())

    size = pl.col('size')
    b = flows.grid.with_columns(
        pl.when(among(bounded)).then(size * pl.col('relative_rate_min')).otherwise(0.0).alias('rate_min'),
        pl.when(among(bounded)).then(size * pl.col('relative_rate_max')).otherwise(np.inf).alias('rate_max'),
    )
    pinned = among(profiled) & pl.col('fixed').is_not_null()
    b = b.with_columns(
        pl.when(pinned).then(size * pl.col('fixed')).otherwise(pl.col('rate_min')).alias('rate_min'),
        pl.when(pinned).then(size * pl.col('fixed')).otherwise(pl.col('rate_max')).alias('rate_max'),
    )
    # A decided size is a variable: the rate is free here, and the envelope
    # holds against the size in rows.
    free = among(sizing_ids | set(invest_ids))
    b = b.with_columns(
        pl.when(free).then(0.0).otherwise(pl.col('rate_min')).alias('rate_min'),
        pl.when(free).then(np.inf).otherwise(pl.col('rate_max')).alias('rate_max'),
    )
    # A gated flow's rate is carried by `running`, unless a decided size takes over.
    gated = among(own_ids)
    scaled_max = size * pl.when(among(profiled)).then(pl.col('fixed')).otherwise(pl.col('relative_rate_max'))
    b = b.with_columns(
        pl.when(gated).then(0.0).otherwise(pl.col('rate_min')).alias('rate_min'),
        pl.when(gated & among(sizing_ids))
        .then(np.inf)
        .when(gated)
        .then(scaled_max)
        .otherwise(pl.col('rate_max'))
        .alias('rate_max'),
    )
    for key in ('rate_min', 'rate_max'):
        sources[key] = b.select(['flow', 'time', 'period', pl.col(key).alias('value')])
    # The envelope spans every flow a size scales: decided sizes, and gated flows.
    scaled = flows.grid.filter(among(sizing_ids | set(invest_ids) | set(status_of)))
    grid_keys = ['flow', 'time', 'period']
    sources['relative_rate_max'] = _column(scaled, grid_keys, 'relative_rate_max', drop_zero=True)
    sources['relative_rate_min'] = _column(scaled, grid_keys, 'relative_rate_min', drop_zero=True)
    sources['fixed_relative_profile'] = _column(scaled, grid_keys, 'fixed', drop_zero=True)
    sources['is_bounded'] = _flags('flow', bounded)
    sources['is_profile'] = _flags('flow', flows.profiled)
    sources['size_bound'] = _size_bound(flows)
    sources |= _sizing(flows, horizon)

    # --- carriers, conversion, storage, effects -------------------------------
    sign = {bf.id: float(bf.sign) for bf in bound}
    sources['carrier_sign'] = _frame({'flow': ids, 'value': [sign[f] for f in ids]}, {'flow': _STR, 'value': _FLOAT})
    linear, converter_of, width = _converters(converters, horizon)
    curves, curve_converter_of, bp_width = _piecewise(converters, horizon)
    sources |= linear | curves | _storages(storages, horizon)
    effect_ids = [e.id for e in effects]
    sources |= _effects(effects, objective_weights(effect_ids, objective), horizon)
    sources['dt'] = pl.DataFrame({'time': horizon.time_series, 'value': horizon.dt})
    sources['time_weight'] = pl.DataFrame({'time': horizon.time_series, 'value': np.ones(horizon.n_time)})

    converter_of |= curve_converter_of

    def relation(mapping: dict[str, str], key: str, target: str) -> pl.DataFrame:
        pairs = [(k, mapping[k]) for k in ids if k in mapping]
        return pl.DataFrame(pairs, schema={key: _STR, target: _STR}, orient='row')

    carrier_of = {
        bf.id: node_id(bf.flow.carrier, bf.flow.node) if bf.flow.node is not None else bf.flow.carrier for bf in bound
    }
    sources['carrier_of'] = relation(carrier_of, 'flow', 'carrier')
    sources['converter_of'] = relation(converter_of, 'flow', 'converter')
    sources['status_of'] = relation(status_of, 'flow', 'status_entity')

    carrier_ids = [node_id(c.id, n) if n else c.id for c in carriers for n in c.nodes or [None]]
    converter_ids = list(
        dict.fromkeys(
            [c.id for c in converters if c.conversion is None] + [c.id for c in converters if c.conversion is not None]
        )
    )

    def axis(name: str, values: Any, dtype: Any) -> pl.DataFrame:
        return pl.DataFrame({name: list(values)}, schema={name: dtype})

    return sources | {
        'time': pl.DataFrame({'time': horizon.time_series}),
        'period': axis('period', horizon.periods, _INT),
        'build_period': axis('build_period', horizon.periods, _INT),
        'flow': axis('flow', ids, _STR),
        'carrier': axis('carrier', carrier_ids, _STR),
        'converter': axis('converter', converter_ids, _STR),
        'eq_idx': axis('eq_idx', range(width), _INT),
        'storage': axis('storage', [s.id for s in storages], _STR),
        'effect': axis('effect', effect_ids, _STR),
        'source': axis('source', effect_ids, _STR),
        'status_entity': axis('status_entity', [i for i, _ in entities], _STR),
        'side': axis('side', ['charge', 'discharge'], _STR),
        'state': axis('state', ['on', 'off'], _STR),
        'bp': axis('bp', range(bp_width), _INT),
    }
