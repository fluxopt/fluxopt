"""Bind a :class:`~fluxopt.model_data.ModelData` to fluxopt's math program.

The data half of the build: this module emits the parameter tables
:data:`PROGRAM` declares, and specsolve does the rest.

Sparsity is carried by *row absence* — a parameter keeps its declared rank
while its table holds only live entries. Arrays at or below a variable's own
grid (bounds) stay dense; only the ones whose rank exceeds it
(``effects_per_flow_hour``, ``conversion_factor``) are filtered, which is where the size is.

Every relation between entities is a coordinate on the ``flow`` dimension
rather than a matrix, so topology travels as rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from fluxopt.leontief import leontief

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData

#: The directory holding fluxopt's math, one YAML fragment per feature.
#: Shipped as package data.
PROGRAM = Path(__file__).with_name('program')


def program(time: str = 'datetime') -> Any:
    """fluxopt's math, composed from its fragments, loaded and checked.

    A :class:`mathspec.Spec`. Each file under :data:`PROGRAM` states one
    feature and loads on its own; ``effects.yaml`` declares the two halves of
    the ledger as sums, and every feature adds its own term to them, so
    ``merge`` writes the ledger. The engine verbs come from specsolve.

    Args:
        time: The dtype of the ``time`` dimension, which the fragments declare
            as ``datetime``: ``int`` for a system whose steps are numbered.
            A system's own is ``data.dims.time_dtype``.
    """
    from mathspec import merge

    declared = '  time: {dtype: datetime}\n'
    fragments = {
        path.stem: path if time == 'datetime' else path.read_text().replace(declared, f'  time: {{dtype: {time}}}\n')
        for path in sorted(PROGRAM.glob('*.yaml'))
    }
    return merge(fragments, description='fluxopt: flows, converters and storages, and what they cost.')


#: Parameters the YAML declares with a `period` axis *and* emit without one.
#: Anything here is cross-joined onto every period. A frame-backed table
#: carries its own period column and is not in this list.
PERIOD_PARAMS = frozenset(
    {
        'prior_level',
        'pw_avail_bound',
        'flow_hours_min',
        'flow_hours_max',
        'load_factor_min',
        'load_factor_max',
        'lifetime_window',
        'prior_capacity_active',
    }
)

#: Parameters carrying the `build_period` axis; it is put on periods the same way.
#: The at-build coefficient tables are frame-backed and already carry theirs.
BUILD_PERIOD_PARAMS = frozenset({'lifetime_window'})


def _live(frame: pl.DataFrame, value: pl.Expr, *, drop_zero: bool = True) -> pl.DataFrame:
    """A `(flow, time, period, value)` table from one expression over *frame*.

    Nulls always go: a row the expression could not compute is a coefficient
    nobody declared. Zeros go too unless the parameter also stands on a
    constant side, where a dropped zero would read as a bound rather than as
    an absent coefficient.
    """
    out = frame.select(['flow', 'time', 'period', value.alias('value')]).drop_nulls('value')
    return out.filter(pl.col('value') != 0) if drop_zero else out


def _size_upper(data: ModelData) -> dict[str, float]:
    """Static upper bound on each flow's size — fixed value or sizing max.

    Zero for a flow with neither, which is what a flow that cannot carry
    anything is worth as a big-M.
    """
    fds = data.flows
    upper = dict.fromkeys(fds.ids, 0.0)
    if fds.sizing is not None:
        upper.update(zip(fds.sizing.bounds['entity'], fds.sizing.bounds['size_max'], strict=True))
    upper.update(zip(fds.sizes['flow'], fds.sizes['size'], strict=True))
    return upper


#: Index columns the program declares with an integer dtype; every other index
#: column is a string label.
_INT_DIMS = frozenset({'time', 'period', 'build_period', 'eq_idx', 'bp'})

#: Parameters the program declares ``dtype: bool``.
_BOOL_PARAMS = frozenset(
    {
        'conversion_active',
        'pw_bp_present',
        'is_cyclic',
        'is_bounded',
        'is_profile',
        'has_duration',
        'forced_at_start',
        'has_sizing',
        'mandatory',
        'has_invest',
        'has_prior_capacity',
        'prevent_simultaneous',
        'has_capacity_sizing',
        'capacity_mandatory',
    }
)


def _stamp_empty_dtypes(sources: dict[str, Any]) -> None:
    """Give every zero-row table the dtypes its parameter declares.

    Pandas cannot type an empty column and picks ``float64``, which the engine
    reads as a numeric label space and refuses against a string dimension. A
    frame with rows carries its own types and is left alone; a frame without
    any has nothing to carry, so the declared types are stamped on here rather
    than guarded at each of the two dozen places one can be produced —
    ``_tidy`` over an all-masked array and the period re-indexing both make
    them.
    """
    for name, df in sources.items():
        if not isinstance(df, pd.DataFrame) or not df.empty:
            continue
        typed = {c: pd.Series([], dtype='int64' if c in _INT_DIMS else 'object') for c in df.columns if c != 'value'}
        typed['value'] = pd.Series([], dtype='bool' if name in _BOOL_PARAMS else 'float64')
        sources[name] = pd.DataFrame(typed)


def _flags(name: str, dim: str, ids: list[str]) -> pd.DataFrame:
    """A boolean table marking *ids* true — typed even when none qualify.

    The empty case is the one that bites: a list comprehension that filters
    everything out leaves pandas to type the label column, and it picks
    ``float64``. :func:`_empty` is what an absent feature looks like.
    """
    return _empty(name, dim) if not ids else pd.DataFrame({dim: ids, 'value': True})


def _empty(name: str, *index_cols: str) -> pd.DataFrame:
    """An empty table for *name*, carrying the dtypes the program declares.

    A parameter with no live entries still binds, and the engine checks a
    label column against its dimension's own — so an all-empty frame has to
    say what it would have held. Pandas types an empty column ``float64``,
    which reads as a numeric label space and is refused.
    """
    cols = {c: pd.Series([], dtype='int64' if c in _INT_DIMS else 'object') for c in index_cols}
    cols['value'] = pd.Series([], dtype='bool' if name in _BOOL_PARAMS else 'float64')
    return pd.DataFrame(cols)


class UnsupportedFeatureError(RuntimeError):
    """The ModelData uses a feature the program does not express yet.

    Raised rather than silently dropping the feature: a missing constraint
    would still solve, just to the wrong answer.
    """


def _tidy(da: xr.DataArray, *, drop_zero: bool) -> pd.DataFrame:
    """Tidy `(dims..., value)` frame; live rows only when *drop_zero*."""
    vals = da.values
    # NaN means "absent"; +/-inf is a legitimate bound and must survive
    keep = ~np.isnan(vals) if vals.dtype.kind == 'f' else np.ones(vals.shape, dtype=bool)
    if drop_zero and vals.dtype.kind == 'f':
        keep = keep & (vals != 0)
    idx = np.nonzero(keep)
    cols: dict[str, Any] = {}
    for dim, positions in zip(da.dims, idx, strict=True):
        labels = da.coords[dim].values[positions] if dim in da.coords else positions
        cols[str(dim)] = labels
    cols['value'] = vals[keep]
    return pd.DataFrame(cols)


def _with_time_ordinals(frame: pl.DataFrame, dims: Any) -> pl.DataFrame:
    """Replace a frame's ``time`` labels with the positions the data layer uses.

    Most containers index time by position, and a few carry the timestamp
    the element layer was given; this puts the few on the same positions so
    the binder joins them alike. `_labelled` puts every table back on the
    labels once they are joined.

    Both sides are cast to one time unit first. A freshly built frame carries
    microseconds and one read back from netCDF carries nanoseconds, and polars
    refuses to join across the two — which is the good outcome, since the
    alternative is the failure the labels themselves have: numpy datetimes are
    nanoseconds and reading their raw integers as microseconds turns 2024 into
    the year 55969, so the join matches nothing rather than failing.
    """
    unit = pl.Datetime('us')
    labels = pd.to_datetime(dims.time.values).to_pydatetime().tolist()
    ordinals = pl.DataFrame({'time': labels, 'ord': list(range(len(labels)))}).with_columns(pl.col('time').cast(unit))
    return frame.with_columns(pl.col('time').cast(unit)).join(ordinals, on='time').drop('time').rename({'ord': 'time'})


def _effect_rows(frame: pl.DataFrame, entity: str, column: str, *axes: str) -> pl.DataFrame:
    """A coefficient table keyed on *entity*, its live rows only.

    The containers key on `entity` because they serve flows and storages
    alike; the parameter a table feeds names the one it is for.
    """
    return frame.select([pl.col('entity').alias(entity), 'effect', *axes, pl.col(column).alias('value')]).filter(
        pl.col('value') != 0
    )


def _chained(cf: xr.DataArray | None, dims: Any) -> pl.DataFrame:
    """What one unit charged to an effect adds to every other, through every chain.

    ``(I - C)^-1 - I``, so the ledger reads each effect as its own charge plus
    shares of the others' and needs no variable per effect to do it. The
    diagonal is zero because a cycle is refused, which keeps the table as
    sparse as the chains are.
    """
    schema = pl.Schema({'effect': pl.String(), 'source': pl.String(), 'period': pl.Int64(), 'value': pl.Float64()})
    if cf is None:
        return pl.DataFrame(schema=schema)
    ids = cf.coords['effect'].values
    identity = xr.DataArray(
        np.eye(len(ids)), dims=['effect', 'source_effect'], coords={'effect': ids, 'source_effect': ids}
    )
    rows = pl.from_pandas(_tidy(leontief(cf) - identity, drop_zero=True)).rename({'source_effect': 'source'})
    if 'period' not in rows.columns:
        rows = rows.join(
            pl.DataFrame({'period': list(range(dims.n_periods))}, schema={'period': pl.Int64}), how='cross'
        )
    return rows.select(list(schema)).cast(schema)


def _labelled(table: Any, by_position: dict[str, tuple[dict[int, Any], Any]]) -> Any:
    """The table with each position on ``time`` and the period axes replaced by its label.

    The tables are built on positions, because that is how the data layer
    indexes; the program is bound on the labels the user wrote, so what a
    caller reads or edits is the timestamp and the year.
    """
    if not isinstance(table, pl.DataFrame | pd.DataFrame):
        return table
    frame = table if isinstance(table, pl.DataFrame) else pl.from_pandas(table)
    axes = [axis for axis in by_position if axis in frame.columns]
    if not axes:
        return table
    frame = frame.with_columns(
        pl.col(axis).replace_strict(by_position[axis][0], return_dtype=by_position[axis][1]) for axis in axes
    )
    return frame if isinstance(table, pl.DataFrame) else frame.to_pandas()


def _reject_unsupported(data: ModelData) -> None:
    fds = data.flows
    if data.piecewise is not None:
        bad = sorted(set(data.piecewise.curves['method'].to_list()) & {'lp'})
        if bad:
            raise UnsupportedFeatureError(
                "piecewise method 'lp' states a curve as its segment lines, which this lane has "
                'no formulation for — the one it does have interpolates between breakpoints, so it '
                'would answer a different question. Use the default method. lpspec has it as '
                '`method: lp` since #926, but only through its `piecewise:` block, which takes a '
                'static list of links and so cannot state a curve whose arity is data.'
            )
    if fds.invest is not None and not data.dims.has_periods:
        raise UnsupportedFeatureError('investment requires multi-period optimization (periods must be specified)')
    if fds.sizing is not None and data.status is not None:
        # Component entities never collide with a flow-sizing id, so the
        # intersection picks out exactly the flows carrying their own Status.
        both = set(fds.sizing.ids) & set(data.status.ids)
        profile = both & set(fds.profiled_ids)
        if profile:
            # fluxopt rejects this combination too — no formulation exists.
            raise UnsupportedFeatureError(f'fixed profile with status+sizing has no formulation: {sorted(profile)}')


def build_sources(data: ModelData, objective: dict[str, float]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Emit the parameter tables and coordinates for :data:`PROGRAM`.

    Args:
        data: The model data to bind. Both backends read the same object.
        objective: Effect ids mapped to their objective weight, as
            :func:`~fluxopt.math.solve.solve` takes it.

    Returns:
        ``(sources, coords)`` ready to pass to ``lpspec.solve``.

    Raises:
        UnsupportedFeatureError: If *data* uses a feature the program does
            not express yet, rather than dropping it silently.
    """
    _reject_unsupported(data)
    fds, dims = data.flows, data.dims

    ordinals = dims.timesteps['time'].to_list()
    dt_by_time = dims.timesteps.select(['time', 'dt'])
    size_upper_of = _size_upper(data)

    flow_ids = fds.ids
    sources: dict[str, Any] = {}
    # Bound up front: the feature blocks below fill these only when the
    # corresponding container is present, but later blocks read them.
    sizing_ids: list[str] = []
    status_ids: list[str] = []
    # Lump-domain accumulators, filled by the flow- and storage-sizing blocks.
    #: (parameter, entity dim, frame, value column)
    lump_frames: list[tuple[str, str, pl.DataFrame, str]] = []
    #: (parameter, frame, value column) — charged where the build happened
    at_build_frames: list[tuple[str, pl.DataFrame, str]] = []

    # --- flow rate bounds: dense, they sit at the variable's own grid -----
    # The envelope carries the relative pair, the profile and the fixed size
    # join onto it, and the bound is then one chain of overrides read top to
    # bottom — the same order the array version applied them in.
    envelope = fds.envelope
    profile = fds.fixed_profile.rename({'value': 'fixed'})
    grid = envelope.join(fds.sizes, on='flow', how='left').join(profile, on=['flow', 'time', 'period'], how='left')

    sz, inv, st = fds.sizing, fds.invest, data.status
    sizing_ids = sz.ids if sz is not None else []
    invest_ids = inv.ids if inv is not None else []
    # The status axis holds both kinds; these are the flows whose own rate
    # bounds the binary takes over, so components have no business here.
    status_ids = [e for e in st.ids if e in set(flow_ids)] if st is not None else []
    is_bounded, is_profile = fds.bounded_ids, fds.profiled_ids

    def among(ids: list[str]) -> pl.Expr:
        """Rows whose flow is one of *ids*."""
        return pl.col('flow').is_in(pl.Series(ids, dtype=pl.String).implode())

    scaled_max = pl.col('size') * pl.when(among(is_profile)).then(pl.col('fixed')).otherwise(
        pl.col('relative_rate_max')
    )
    bounds = grid.with_columns(
        pl.when(among(is_bounded)).then(pl.col('size') * pl.col('relative_rate_min')).otherwise(0.0).alias('rate_min'),
        pl.when(among(is_bounded))
        .then(pl.col('size') * pl.col('relative_rate_max'))
        .otherwise(np.inf)
        .alias('rate_max'),
    )
    # A fixed profile pins the rate: both bounds land on the same value.
    pinned = among(is_profile) & pl.col('fixed').is_not_null()
    bounds = bounds.with_columns(
        pl.when(pinned).then(pl.col('size') * pl.col('fixed')).otherwise(pl.col('rate_min')).alias('rate_min'),
        pl.when(pinned).then(pl.col('size') * pl.col('fixed')).otherwise(pl.col('rate_max')).alias('rate_max'),
    )
    # An optimized size is a variable, so the rate is free here and the
    # envelope is applied against the size variable instead.
    free = among([*sizing_ids, *invest_ids])
    bounds = bounds.with_columns(
        pl.when(free).then(0.0).otherwise(pl.col('rate_min')).alias('rate_min'),
        pl.when(free).then(np.inf).otherwise(pl.col('rate_max')).alias('rate_max'),
    )
    # `on` carries the envelope for status flows; the variable itself is free
    # above 0, except when it is sized too and the size variable takes over.
    gated = among(status_ids)
    bounds = bounds.with_columns(
        pl.when(gated).then(0.0).otherwise(pl.col('rate_min')).alias('rate_min'),
        pl.when(gated & among(sizing_ids))
        .then(np.inf)
        .when(gated)
        .then(scaled_max)
        .otherwise(pl.col('rate_max'))
        .alias('rate_max'),
    )
    for key in ('rate_min', 'rate_max'):
        sources[key] = bounds.select(['flow', 'time', 'period', pl.col(key).alias('value')])
    # The big-M for every binary that has to release a rate — a ramp across a
    # start-up, a storage's charge/discharge exclusion. Stated once on `flow`;
    # the storage side reads it through `port_of` rather than keeping a copy
    # on its own axis.
    sources['size_bound'] = pd.DataFrame({'flow': flow_ids, 'value': [size_upper_of[f] for f in flow_ids]})

    # --- carrier balance --------------------------------------------------
    membership = data.carriers.membership
    flow_index = pd.DataFrame({'flow': flow_ids, 'carrier_of': membership['carrier'].to_list()})
    sources['carrier_sign'] = pd.DataFrame({'flow': flow_ids, 'value': membership['sign'].to_numpy()})

    # --- converters -------------------------------------------------------
    if data.converters is not None:
        cds = data.converters
        coeffs = cds.coefficients
        conv_of = dict(zip(coeffs['flow'], coeffs['converter'], strict=True))
        flow_index['converter_of'] = [conv_of.get(f) for f in flow_ids]
        # Already the table the parameter wants, keyed by timestamp where
        # the rest of the data layer uses positions.
        sources['conversion_factor'] = _with_time_ordinals(coeffs.filter(pl.col('value') != 0), dims).select(
            ['flow', 'eq_idx', 'time', 'value']
        )
        # One row per equation each converter states — the counts, expanded.
        sources['conversion_active'] = pl.DataFrame(
            {
                'converter': [c for c, n in zip(cds.ids, cds.equations['n_equations'], strict=True) for _ in range(n)],
                'eq_idx': [i for n in cds.equations['n_equations'] for i in range(n)],
            }
        ).with_columns(pl.lit(True).alias('value'))
    else:
        flow_index['converter_of'] = None
        sources['conversion_factor'] = _empty('conversion_factor', 'flow', 'eq_idx', 'time')
        sources['conversion_active'] = _empty('conversion_active', 'converter', 'eq_idx')

    # --- storage ----------------------------------------------------------
    storage_ids: list[str] = []
    if data.storages is not None:
        sds = data.storages
        storage_ids = sds.ids
        port_of = pl.concat(
            [
                sds.storages.select(
                    pl.col(column).alias('flow'), pl.col('storage'), pl.lit(side).alias('side')
                ).drop_nulls('flow')
                for side, column in (('charge', 'charge_flow'), ('discharge', 'discharge_flow'))
            ]
        )

        # One join carries every per-timestep storage parameter, since they
        # all live on the same (storage, time) rows.
        profiles = _with_time_ordinals(sds.profiles, dims).join(dt_by_time, on='time')

        sources['retention'] = profiles.select(
            ['storage', 'time', ((1 - pl.col('loss')) ** pl.col('dt')).alias('value')]
        )
        sources['storage_coeff'] = pl.concat(
            [
                profiles.select(['storage', pl.lit(side).alias('side'), 'time', value.alias('value')])
                for side, value in (
                    ('charge', pl.col('eta_charge') * pl.col('dt')),
                    ('discharge', -pl.col('dt') / pl.col('eta_discharge')),
                )
            ]
        )

        # An absent capacity row is a storage whose capacity is a variable, so
        # its absolute level bounds are not knowable here: 0 and infinity are
        # what the program reads while the relative pair does the bounding.
        absolute = profiles.join(sds.capacity, on='storage', how='left')
        sources['level_min'] = absolute.select(
            ['storage', 'time', (pl.col('relative_level_min') * pl.col('capacity')).fill_null(0.0).alias('value')]
        )
        sources['level_max'] = absolute.select(
            ['storage', 'time', (pl.col('relative_level_max') * pl.col('capacity')).fill_null(np.inf).alias('value')]
        )
        sources['given_capacity'] = sds.capacity.select(['storage', pl.col('capacity').alias('value')])
        csz = sds.sizing
        if csz is not None:
            cap_ids = csz.ids
            sources['has_capacity_sizing'] = _flags('has_capacity_sizing', 'storage', cap_ids)
            sources['capacity_mandatory'] = _flags(
                'capacity_mandatory', 'storage', csz.bounds.filter('mandatory')['entity'].to_list()
            )
            sources['capacity_min'] = pd.DataFrame({'storage': cap_ids, 'value': csz.bounds['size_min'].to_numpy()})
            sources['capacity_max'] = pd.DataFrame({'storage': cap_ids, 'value': csz.bounds['size_max'].to_numpy()})
            sized = profiles.filter(pl.col('storage').is_in(pl.Series(cap_ids).implode()))
            for key, column in (
                ('relative_level_min', 'relative_level_min'),
                ('relative_level_max', 'relative_level_max'),
            ):
                sources[key] = sized.select(['storage', 'time', pl.col(column).alias('value')]).filter(
                    pl.col('value') != 0
                )
            lump_frames += [
                ('effects_per_capacity', 'storage', csz.effects, 'per_size'),
                ('effects_fixed_capacity', 'storage', csz.effects, 'fixed'),
            ]
        for key, column in (('is_cyclic', 'cyclic'), ('prevent_simultaneous', 'prevent_simultaneous')):
            sources[key] = _flags(key, 'storage', sds.storages.filter(pl.col(column))['storage'].to_list())
        # `prior_level` is dense: it sits on the constant side of the first
        # step's balance, where an absent row is a binding zero — which is
        # also what an unset prior level means.
        sources['prior_level'] = (
            sds.storages.select('storage')
            .join(sds.levels.select(['storage', pl.col('prior_level').alias('value')]), on='storage', how='left')
            .with_columns(pl.col('value').fill_null(0.0))
        )
        for key in ('final_level_min', 'final_level_max'):
            sources[key] = sds.levels.select(['storage', pl.col(key).alias('value')]).drop_nulls()
    else:
        port_of = pl.DataFrame(schema={'flow': pl.String, 'storage': pl.String, 'side': pl.String})
        for name, dcols in (
            ('is_cyclic', ['storage']),
            ('prior_level', ['storage']),
            ('final_level_min', ['storage']),
            ('final_level_max', ['storage']),
            ('prevent_simultaneous', ['storage']),
            ('given_capacity', ['storage']),
        ):
            sources[name] = pd.DataFrame({c: [] for c in [*dcols, 'value']})
        for name, dcols in (
            ('storage_coeff', ['storage', 'side', 'time']),
            ('retention', ['storage', 'time']),
            ('level_min', ['storage', 'time']),
            ('level_max', ['storage', 'time']),
        ):
            sources[name] = pd.DataFrame({c: [] for c in [*dcols, 'value']})

    # --- status -----------------------------------------------------------
    # One table, one axis. An entity here is a flow carrying its own on/off
    # decision or a component whose decision governs several flows; they carry
    # the same fields and obey the same math, so the program states the family
    # once over `status_entity`. Which rows read which binary is `status_of`,
    # and nothing else distinguishes them.
    #: (parameter, frame, value column) — status coefficients, per timestep
    status_effect_frames: list[tuple[str, pl.DataFrame, str]] = []
    #: flow id -> the entity whose binary gates it. A self-status flow maps to
    #: itself; a governed flow to its component; an ungated flow to nothing.
    status_of: dict[str, str] = {f: f for f in status_ids}
    # A piecewise curve's flows are gated by its convexity row, which already
    # pins every weight to zero when the binary is off. Gating them a second
    # time per flow would be redundant, and wrong for the links the curve
    # only bounds.
    pw_comps = set(data.piecewise.converter_ids()) if data.piecewise is not None else set()
    status_of.update(
        {
            fid: owner
            for fid, owner in zip(fds.governed_by['flow'], fds.governed_by['component'], strict=True)
            if owner not in pw_comps
        }
    )

    flow_index['status_of'] = [status_of.get(f) for f in flow_ids]
    entity_ids: list[str] = st.ids if st is not None else []
    gated_ids = [f for f in flow_ids if f in status_of]

    sources['is_bounded'] = _flags('is_bounded', 'flow', is_bounded)
    sources['is_profile'] = _flags('is_profile', 'flow', is_profile)

    if st is not None:
        durations, prior, status_effects = st.durations, st.prior, st.effects
        all_entities = pl.DataFrame({'status_entity': entity_ids}, schema={'status_entity': pl.String})

        def per_entity(frame: pl.DataFrame, column: str) -> Any:
            """One column of a frame, keyed on the entity axis, live rows only."""
            live = frame.filter(pl.col(column).is_not_null()).select(
                [pl.col('entity').alias('status_entity'), pl.col(column).alias('value')]
            )
            return live if len(live) else _empty(column, 'status_entity')

        horizon = float(dims.timesteps['dt'].sum())
        states = (('on', 'uptime'), ('off', 'downtime'))

        def per_state(frame: pl.DataFrame, column: str) -> pl.DataFrame:
            """One `{kind}_…` column pair of a wide frame, long over `state`, live rows only."""
            return pl.concat(
                [
                    frame.select(
                        pl.col('entity').alias('status_entity'),
                        pl.lit(state).alias('state'),
                        pl.col(column.format(kind=kind)).cast(pl.Float64).alias('value'),
                    )
                    for state, kind in states
                ]
            ).drop_nulls('value')

        bounds_long = per_state(durations, '{kind}_min').join(
            per_state(durations, '{kind}_max'), on=['status_entity', 'state'], how='full', coalesce=True, suffix='_max'
        )
        sources['has_duration'] = bounds_long.select(['status_entity', 'state', pl.lit(True).alias('value')])
        sources['duration_min'] = per_state(durations, '{kind}_min')
        sources['initial_status'] = per_entity(prior, 'initial')
        sources['previous_duration'] = per_state(prior, 'previous_{kind}')

        # Big-M covers the whole horizon plus whatever ran before it. An
        # entity with no prior has no row in `prior`, and no prior is zero.
        big_m = (
            all_entities.join(pl.DataFrame({'state': [st_ for st_, _ in states]}), how='cross')
            .join(sources['previous_duration'], on=['status_entity', 'state'], how='left')
            .select(['status_entity', 'state', (pl.col('value').fill_null(0.0) + horizon).alias('value')])
        )
        sources['duration_big_m'] = big_m

        # The duration counter's ceiling: the declared maximum where there is
        # one, the big-M where there is not.
        sources['duration_upper'] = (
            bounds_long.select(['status_entity', 'state', 'value_max'])
            .join(big_m, on=['status_entity', 'state'])
            .select(['status_entity', 'state', pl.col('value_max').fill_null(pl.col('value')).alias('value')])
        )

        # A prior stay shorter than the minimum forces continuation.
        sources['forced_at_start'] = (
            sources['previous_duration']
            .join(sources['duration_min'], on=['status_entity', 'state'], suffix='_min')
            .filter((pl.col('value') > 0) & (pl.col('value') < pl.col('value_min')))
            .select(['status_entity', 'state', pl.lit(True).alias('value')])
        )

        # `dt` turns a per-running-hour rate into the step's cost; a startup
        # happens once at the step, so it is not scaled.
        scaled = (
            _with_time_ordinals(status_effects, dims)
            .join(dt_by_time, on='time')
            .with_columns((pl.col('running') * pl.col('dt')).alias('running'))
        )
        status_effect_frames += [
            ('effects_per_running_hour', scaled, 'running'),
            ('effects_per_startup', scaled, 'startup'),
        ]
    else:
        sources['initial_status'] = _empty('initial_status', 'status_entity')
        for n in (
            'has_duration',
            'duration_min',
            'duration_upper',
            'duration_big_m',
            'previous_duration',
            'forced_at_start',
        ):
            sources[n] = _empty(n, 'status_entity', 'state')

    sources['dt'] = dims.timesteps.select(['time', pl.col('dt').alias('value')])

    # --- sizing -----------------------------------------------------------
    if sz is not None:
        bounds = sz.bounds
        sources['has_sizing'] = _flags('has_sizing', 'flow', sizing_ids)
        sources['mandatory'] = _flags('mandatory', 'flow', bounds.filter('mandatory')['entity'].to_list())
        sources['size_min'] = pd.DataFrame({'flow': sizing_ids, 'value': bounds['size_min'].to_numpy()})
        sources['size_max'] = pd.DataFrame({'flow': sizing_ids, 'value': bounds['size_max'].to_numpy()})
        lump_frames += [
            ('effects_per_size', 'flow', sz.effects, 'per_size'),
            ('effects_fixed', 'flow', sz.effects, 'fixed'),
        ]
    else:
        for n in ('has_sizing', 'mandatory', 'size_min', 'size_max'):
            sources[n] = _empty(n, 'flow')

    if 'has_capacity_sizing' not in sources:
        for n in ('has_capacity_sizing', 'capacity_mandatory', 'capacity_min', 'capacity_max'):
            sources[n] = _empty(n, 'storage')
        for n in ('relative_level_min', 'relative_level_max'):
            sources[n] = _empty(n, 'storage', 'time')

    # --- ramps ------------------------------------------------------------
    # A ramp limit is per hour, so the step's allowance is limit x dt, per
    # unit of size; the program multiplies by `flow_size`.
    ramps = fds.ramps.join(dt_by_time, on='time')
    for kind in ('up', 'down'):
        # A ramp of 0 is a row: the row is what says the flow has a ramp.
        declared = ramps.filter(pl.col(f'ramp_{kind}').is_not_null())
        sources[f'ramp_{kind}_coeff'] = _live(declared, pl.col(f'ramp_{kind}') * pl.col('dt'), drop_zero=False)
    # --- investment -------------------------------------------------------
    if inv is not None:
        period_labels_inv: list[Any] = dims.periods['label'].to_list()
        n_p = len(period_labels_inv)
        # No row means forever, so the lookup's default is the absence.
        expires = dict(zip(inv.lifetime['entity'], inv.lifetime['periods'], strict=True))
        lifetime = [expires.get(f) for f in invest_ids]
        prior = inv.bounds['prior_size'].to_numpy()
        window = np.zeros((len(invest_ids), n_p, n_p))
        prior_active = np.zeros((len(invest_ids), n_p))
        for f_idx in range(len(invest_ids)):
            lt_int = lifetime[f_idx]
            for p_idx in range(n_p):
                for b_idx in range(n_p):
                    alive = b_idx <= p_idx if lt_int is None else b_idx <= p_idx < b_idx + lt_int
                    window[f_idx, p_idx, b_idx] = float(alive)
                if prior[f_idx] > 0 and (lt_int is None or p_idx < lt_int):
                    prior_active[f_idx, p_idx] = 1.0
        coords_w = {'flow': invest_ids, 'period': period_labels_inv, 'build_period': period_labels_inv}
        sources['lifetime_window'] = _tidy(
            xr.DataArray(window, dims=['flow', 'period', 'build_period'], coords=coords_w), drop_zero=True
        )
        sources['prior_capacity_active'] = _tidy(
            xr.DataArray(
                prior_active, dims=['flow', 'period'], coords={'flow': invest_ids, 'period': period_labels_inv}
            ),
            drop_zero=False,
        )
        sources['has_invest'] = _flags('has_invest', 'flow', invest_ids)
        # The same parameter the sizing block fills. A flow's size is a
        # `Sizing` or an `Investment`, never both, so the two never collide.
        sources['mandatory'] = pd.concat(
            [sources['mandatory'], pd.DataFrame({'flow': invest_ids, 'value': inv.bounds['mandatory'].to_numpy()})],
            ignore_index=True,
        )
        # The same two parameters the sizing block fills, over the other half
        # of the sized flows.
        for key in ('size_min', 'size_max'):
            sources[key] = pd.concat(
                [sources[key], pd.DataFrame({'flow': invest_ids, 'value': inv.bounds[key].to_numpy()})],
                ignore_index=True,
            )
        sources['prior_capacity'] = pd.DataFrame({'flow': invest_ids, 'value': prior})
        sources['has_prior_capacity'] = _flags(
            'has_prior_capacity', 'flow', [f for f, ps in zip(invest_ids, prior, strict=True) if ps > 0]
        )
        lump_frames += [
            ('effects_per_size_recurring', 'flow', inv.effects, 'per_size_recurring'),
            ('effects_fixed_recurring', 'flow', inv.effects, 'fixed_recurring'),
        ]
        # A one-time cost belongs to the period the build happened in, not to
        # every period the unit is alive — which as rows is the diagonal: the
        # same period twice, rather than a square matrix multiplied by an eye.
        at_build_frames += [
            ('effects_per_size_at_build', inv.effects, 'per_size_at_build'),
            ('effects_fixed_at_build', inv.effects, 'fixed_at_build'),
        ]
    else:
        for name in (
            'has_invest',
            'prior_capacity',
            'has_prior_capacity',
        ):
            sources[name] = _empty(name, 'flow')
        sources['lifetime_window'] = _empty('lifetime_window', 'flow', 'period', 'build_period')
        sources['prior_capacity_active'] = _empty('prior_capacity_active', 'flow', 'period')

    # The envelope is read by both mechanisms and by the status family, so it
    # must span every flow whose rate is scaled by a size: sized flows, whose
    # `size` is a variable, and gated flows, whose `size_bound` is a number.
    sized_ids = [*sizing_ids, *invest_ids]
    scaled = grid.filter(pl.col('flow').is_in(pl.Series([*sized_ids, *gated_ids], dtype=pl.String).implode()))
    sources['relative_rate_max'] = _live(scaled, pl.col('relative_rate_max'))
    sources['fixed_relative_profile'] = _live(scaled, pl.col('fixed'))
    # Dense: the lower bound also stands on the constant side of
    # `status_sizing_rate_min`, where a dropped zero is a bound rather than an
    # absent coefficient — and a flow whose lower bound is zero is the
    # ordinary case, so dropping it would break exactly the common one.
    sources['relative_rate_min'] = _live(scaled, pl.col('relative_rate_min'), drop_zero=True)

    # --- effects: the sparse one -----------------------------------------
    eds = data.effects
    effect_ids = eds.ids
    # dt stays: a per-flow-hour rate times a duration is the step's energy.
    # The aggregation weight does not — the program applies it in the sum,
    # so a named contribution reads as the physical per-step quantity.
    pairs = fds.effect_pairs.join(dt_by_time, on='time').select(
        ['flow', 'effect', 'time', 'period', (pl.col('value') * pl.col('dt')).alias('value')]
    )
    sources['effects_per_flow_hour'] = _effect_rows(pairs.rename({'flow': 'entity'}), 'flow', 'value', 'time', 'period')
    for name, frame, column in status_effect_frames:
        sources[name] = _effect_rows(frame, 'status_entity', column, 'time', 'period')
    for name in ('effects_per_running_hour', 'effects_per_startup'):
        sources.setdefault(name, _empty(name, 'status_entity', 'effect', 'time', 'period'))

    sources['share'] = _chained(eds.cf_matrix(), dims)

    for name, entity_dim, frame, column in lump_frames:
        sources[name] = _effect_rows(frame, entity_dim, column, 'period')
    for name, frame, column in at_build_frames:
        rows = _effect_rows(frame, 'flow', column, 'period')
        sources[name] = rows.with_columns(pl.col('period').alias('build_period'))
    for name in ('effects_per_size', 'effects_fixed'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period'))
    for name in ('effects_per_capacity', 'effects_fixed_capacity'):
        sources.setdefault(name, _empty(name, 'storage', 'effect', 'period'))
    for name in ('effects_per_size_at_build', 'effects_fixed_at_build'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period', 'build_period'))
    for name in ('effects_per_size_recurring', 'effects_fixed_recurring'):
        sources.setdefault(name, _empty(name, 'flow', 'effect', 'period'))
    # Objective weight x period weight, folded into one parameter. Both are
    # per (effect, period), so the fold is a join and the defaults are what a
    # missing row means: no override, then no global weight, then 1.
    global_weights = dims.periods.select(['period', pl.col('weight').alias('global_weight')])
    period_axis = list(range(dims.n_periods))
    grid = (
        pl.DataFrame({'effect': effect_ids}, schema={'effect': pl.String})
        .join(pl.DataFrame({'period': period_axis}, schema={'period': pl.Int64}), how='cross')
        .join(eds.period_weights, on=['effect', 'period'], how='left')
        .join(global_weights, on='period', how='left')
        .with_columns(pl.col('weight').fill_null(pl.col('global_weight')).fill_null(1.0).alias('weight'))
    )
    objective_by_effect = pl.DataFrame(
        {'effect': effect_ids, 'objective': [float(objective.get(e, 0.0)) for e in effect_ids]},
        schema={'effect': pl.String, 'objective': pl.Float64},
    )
    sources['objective_weight'] = (
        grid.join(objective_by_effect, on='effect')
        .with_columns((pl.col('objective') * pl.col('weight')).alias('value'))
        .filter(pl.col('value') != 0)
        .select(['effect', 'period', 'value'])
    )
    # Weights for the across-period sum: per-effect override, else global, else 1.
    sources['period_weight'] = grid.select(['effect', 'period', pl.col('weight').alias('value')])

    # --- effect limits ----------------------------------------------------
    for key, frame, axes in (
        ('periodic_min', eds.periodic, ['effect', 'period']),
        ('periodic_max', eds.periodic, ['effect', 'period']),
        ('total_min', eds.totals, ['effect']),
        ('total_max', eds.totals, ['effect']),
    ):
        sources[key] = frame.filter(pl.col(key).is_not_null()).select([*axes, pl.col(key).alias('value')])

    # --- temporal boundary mask ------------------------------------------
    sources['time_weight'] = dims.timesteps.select(['time', pl.col('weight').alias('value')])

    # --- flow aggregates ------------------------------------------------
    # A load factor bounds the mean rate as a fraction of the size, so it
    # travels as lambda x T and the program multiplies by `flow_size`.
    total_duration = float((dims.timesteps['dt'] * dims.timesteps['weight']).sum())
    aggregates = fds.aggregates
    for name in ('flow_hours_min', 'flow_hours_max'):
        sources[name] = aggregates.select(['flow', pl.col(name).alias('value')]).drop_nulls('value')
    for kind in ('min', 'max'):
        sources[f'load_factor_{kind}'] = aggregates.select(
            ['flow', (pl.col(f'load_factor_{kind}') * total_duration).alias('value')]
        ).drop_nulls('value')

    # --- piecewise conversion ------------------------------------------
    # The curve tables are already the shape the program wants: a link is a
    # row on `flow`, so nothing has to be reshaped into link slots.
    linear_convs = data.converters.ids if data.converters is not None else []
    bp_width = 0
    pw_status_of: dict[str, str | None] = {}
    pw = data.piecewise
    if pw is not None:
        pw_convs = pw.converter_ids()
        links, curves = pw.links, pw.curves
        # A link is a (converter, flow, bound) — the breakpoints it passes
        # through are its rows, so the identity is what remains after dropping
        # the axes a curve varies along.
        identity = links.select(['converter', 'flow', 'bound']).unique(maintain_order=True)

        sources['pw_bp_value'] = _with_time_ordinals(links.filter(pl.col('value') != 0), dims).select(
            ['flow', 'bp', 'time', 'value']
        )

        # Which breakpoints a curve has. Curves of different width share one
        # `bp` axis, so the mask is what stops a weight existing past the end
        # of a narrower one.
        present = links.select(['converter', 'bp']).unique(maintain_order=True).sort(['converter', 'bp'])
        bp_width = int(present['bp'].max() or 0) + 1 if len(present) else 0  # type: ignore[arg-type]
        sources['pw_bp_present'] = present.with_columns(pl.lit(True).alias('value'))

        gated = curves.filter(pl.col('has_status'))['converter'].unique(maintain_order=True).to_list()
        sources['curve_of'] = identity.select(['flow', 'converter'])
        sources['link_sense'] = identity.select(['flow', pl.col('bound').alias('value')])

        # Availability scales the envelope of the reference link — a curve's
        # first, which is what the eager lane bounds too.
        reference = identity.group_by('converter', maintain_order=True).first()
        sources['pw_ref'] = reference.select(['flow', pl.lit(1.0).alias('value')])
        widest = (
            links.join(reference.select(['converter', 'flow']), on=['converter', 'flow'])
            .group_by(['converter', 'time'])
            .agg(pl.col('value').max().alias('widest'))
        )
        sources['pw_avail_bound'] = _with_time_ordinals(
            curves.join(widest, on=['converter', 'time']).with_columns(
                (pl.col('availability') * pl.col('widest')).alias('value')
            ),
            dims,
        ).select(['converter', 'time', 'value'])

        # A gated curve's Status is keyed by the converter's own id, so the
        # lookup maps it to itself — mapping it to None reads as 'no Status'
        # and leaves the curve ungated.
        pw_status_of = {c: c for c in gated}
        of = dict(zip(identity['flow'], identity['converter'], strict=True))
        flow_index['converter_of'] = [of.get(f) or c for f, c in zip(flow_ids, flow_index['converter_of'], strict=True)]
        converter_ids = linear_convs + [c for c in pw_convs if c not in set(linear_convs)]
    else:
        for name, dcols in (
            ('pw_ref', ('flow',)),
            ('pw_bp_value', ('flow', 'bp', 'time')),
            ('pw_avail_bound', ('converter', 'time')),
        ):
            sources[name] = _empty(name, *dcols)
        sources['pw_bp_present'] = _empty('pw_bp_present', 'converter', 'bp')
        sources['curve_of'] = pl.DataFrame(schema={'flow': pl.String, 'converter': pl.String})
        sources['link_sense'] = pl.DataFrame(schema={'flow': pl.String, 'value': pl.String})
        converter_ids = linear_convs

    # A map is its own source key, keyed `(over, into)` and holding only the
    # labels it maps — a flow charging no storage has no row, rather than a
    # null saying so. An index carrying a column named after a lookup over it
    # is refused, so the two facts stay apart all the way down.
    def maps(index: pd.DataFrame, over: str, into: dict[str, str]) -> dict[str, pl.DataFrame]:
        """One table per lookup declared over *over*, from its index columns."""
        return {
            name: pl.DataFrame(
                {over: mapped[over].tolist(), target: mapped[name].tolist()},
                schema={over: pl.String, target: pl.String},
            )
            for name, target in into.items()
            if name in index.columns
            # pandas 3 reads a missing label as NaN, which no string column takes.
            for mapped in (index[index[name].notna()],)
        }

    flow_axis = pl.DataFrame({'flow': flow_ids}, schema={'flow': pl.String})
    lookup_tables = maps(
        flow_index,
        'flow',
        {
            'carrier_of': 'carrier',
            'converter_of': 'converter',
            'status_of': 'status_entity',
        },
    )

    # Single-period models supply a length-1 period so one program serves both.
    period_labels = dims.period_labels
    period_ord = {v: i for i, v in enumerate(period_labels)}
    p_ordinals = list(range(len(period_labels)))

    def onto_periods(df: Any, axis: str) -> Any:
        """Put *df* on the program's period axis, whichever library holds it.

        A table already naming periods has its labels mapped to ordinals; one
        that does not is the same in every period, so it is crossed onto all
        of them. Both shapes arrive in pandas and in polars, and will keep
        doing so until the last container is converted.
        """
        if isinstance(df, pl.DataFrame):
            if axis in df.columns:
                return df.with_columns(pl.col(axis).replace_strict(period_ord, return_dtype=pl.Int64))
            return df.join(pl.DataFrame({axis: p_ordinals}, schema={axis: pl.Int64}), how='cross')
        if axis in df.columns:
            return df.assign(**{axis: [period_ord[v] for v in df[axis]]})
        return df.merge(pd.DataFrame({axis: p_ordinals}), how='cross')

    for name in PERIOD_PARAMS:
        if (df := sources.get(name)) is not None:
            sources[name] = onto_periods(df, 'period')

    for name in BUILD_PERIOD_PARAMS:
        df = sources.get(name)
        if df is not None and 'build_period' in df.columns:
            sources[name] = onto_periods(df, 'build_period')

    def labels(values: Any) -> np.ndarray:
        """A string dimension's labels, carrying their type even when empty.

        A system with no storages still declares the dimension, and its
        lookups are string maps into it. A bare ``[]`` has no dtype to infer,
        binds as a null label space and fails the join against those columns —
        so the labels travel as a string array, which says what the space
        would have held.
        """
        return np.array(list(values), dtype=str)

    def axis(dim: str, values: Any) -> pl.DataFrame:
        """A dimension's labels as a one-column frame.

        A frame rather than a sequence because a strategy slices its sources:
        `solve_over` filters every table that carries the axis it cuts on, and
        a bare list is not a table it can cut. Costing nothing to always do,
        so the shipped sources are sliceable whether or not this build is.
        """
        return pl.DataFrame({dim: list(values)}, schema={dim: pl.Int64})

    time_labels = dims.timesteps['label']
    coords: dict[str, Any] = {
        'time': pl.DataFrame({'time': time_labels}),
        'period': axis('period', period_labels),
        'build_period': axis('build_period', period_labels),
        'flow': flow_axis,
        'carrier': labels(data.carriers.ids),
        # Both kinds: a converter states linear equations, a piecewise curve,
        # or one of each. The axis is the union, or a curve's own converter
        # would not be a coordinate of the dimension its rows are keyed on.
        'converter': pl.DataFrame({'converter': converter_ids}, schema={'converter': pl.String}),
        'eq_idx': axis('eq_idx', range(data.converters.width) if data.converters is not None else []),
        'storage': labels(storage_ids),
        'effect': labels(effect_ids),
        'source': labels(effect_ids),
        'status_entity': labels(entity_ids),
        'side': labels(['charge', 'discharge']),
        'state': labels(['on', 'off']),
        # numpy, not a list: with no piecewise converter the width is 0 and a
        # bare `[]` has no integer type for the join to match.
        'bp': axis('bp', range(bp_width)),
    }
    lookup_tables['pw_status_of'] = pl.DataFrame(
        {'converter': list(pw_status_of), 'status_entity': [pw_status_of[c] for c in pw_status_of]},
        schema={'converter': pl.String, 'status_entity': pl.String},
    ).drop_nulls('status_entity')
    lookup_tables['port_of'] = port_of
    lookup_tables['same'] = pl.DataFrame(
        {'source': effect_ids, 'effect': effect_ids}, schema={'source': pl.String, 'effect': pl.String}
    )
    _stamp_empty_dtypes(sources)
    by_position = {
        'time': (dict(zip(ordinals, time_labels.to_list(), strict=True)), time_labels.dtype),
        'period': (dict(zip(p_ordinals, period_labels, strict=True)), pl.Int64),
        'build_period': (dict(zip(p_ordinals, period_labels, strict=True)), pl.Int64),
    }
    return {name: _labelled(table, by_position) for name, table in sources.items()} | lookup_tables, coords
