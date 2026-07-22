from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self

import numpy as np
import pandas as pd
import xarray as xr

from fluxopt.types import as_dataarray, fast_concat, normalize_timesteps

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    from fluxopt.components import Converter, Port
    from fluxopt.elements import Carrier, Effect, Flow, Investment, Sizing, Status, Storage
    from fluxopt.types import TimeIndex, Timesteps, Variate


@dataclass(frozen=True)
class _EffectTemplate:
    """Pre-computed shape/dims/coords for an effect-dimensioned zero array."""

    shape: tuple[int, ...]
    dims: tuple[str, ...]
    coords: dict[str, Any]
    as_da_coords: dict[str, Any]

    def zeros(self) -> xr.DataArray:
        """Create a zero-filled DataArray with this template's shape."""
        return xr.DataArray(np.zeros(self.shape), dims=list(self.dims), coords=self.coords)


def _effect_template(
    base_dims: dict[str, Any],
    period: pd.Index | None = None,
) -> _EffectTemplate:
    """Build template shape/dims/coords for effect arrays with optional period.

    Args:
        base_dims: Ordered mapping of dim_name -> coord_values.
        period: Period index to append as trailing dimension.
    """
    dims = list(base_dims.keys())
    coords = dict(base_dims)
    shape = [len(v) for v in base_dims.values()]
    as_da_coords: dict[str, Any] = {k: v for k, v in base_dims.items() if k not in ('effect', 'source_effect')}

    if period is not None:
        dims.append('period')
        coords['period'] = period
        shape.append(len(period))
        as_da_coords['period'] = period

    return _EffectTemplate(
        shape=tuple(shape),
        dims=tuple(dims),
        coords=coords,
        as_da_coords=as_da_coords,
    )


_NC_GROUPS = {
    'flows': 'model/flows',
    'carriers': 'model/carriers',
    'converters': 'model/conv',
    'effects': 'model/effects',
    'storages': 'model/stor',
    'piecewise': 'model/pw',
}


def _raise_netcdf_read_error(path: Path, exc: OSError) -> NoReturn:
    """Re-raise a netCDF read failure, clarifying the Windows non-ASCII path bug.

    netcdf4/libnetcdf (through 4.9.3) fails to open files under non-ASCII
    *directories* on Windows with a misleading ``PermissionError``/``OSError``
    (upstream bug Unidata/netcdf4-python#1482). When the failing path is
    non-ASCII on Windows we surface an actionable message; otherwise the original
    error propagates unchanged. Only read paths are wrapped — the error only
    surfaces if netcdf4 actually raises, so nothing that would work is blocked.

    Args:
        path: The path being read.
        exc: The error raised by the netCDF engine.

    Raises:
        ValueError: On Windows when the failing path contains non-ASCII characters.
        OSError: The original error, on any other platform or path.
    """
    if os.name == 'nt' and not str(path).isascii():
        raise ValueError(
            f'Failed to read netCDF at a path containing non-ASCII characters on Windows: {path}. '
            'netcdf4 cannot open files under non-ASCII directories on Windows '
            '(upstream bug Unidata/netcdf4-python#1482). Use an ASCII-only directory and file name.'
        ) from exc
    raise exc


def _list_groups(nc: Any) -> set[str]:
    """Full paths of all (nested) groups in an open ``netCDF4.Dataset``."""
    out: set[str] = set()

    def walk(group: Any, prefix: str) -> None:
        for name, child in group.groups.items():
            path = f'{prefix}/{name}'
            out.add(path)
            walk(child, path)

    walk(nc, '')
    return out


def _load_group(nc: Any, path: str) -> xr.Dataset:
    """Load one group as a Dataset through an already-open ``netCDF4.Dataset``."""
    from xarray.backends import NetCDF4DataStore

    return xr.open_dataset(NetCDF4DataStore(nc[path])).load()


def _to_dataset(obj: DataclassInstance) -> xr.Dataset:
    """Convert a data dataclass to an xr.Dataset.

    Args:
        obj: Dataclass with DataArray fields and scalar attrs.
    """
    data_vars: dict[str, xr.DataArray] = {}
    attrs: dict[str, object] = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if val is None or isinstance(val, dict):
            continue  # dict fields serialize as netCDF child groups, not variables
        if isinstance(val, xr.DataArray):
            data_vars[f.name] = val
        else:
            attrs[f.name] = val
    ds = xr.Dataset(data_vars)
    ds.attrs.update(attrs)
    return ds


def _dict_field_names(cls: type) -> list[str]:
    """Names of a table dataclass's dict-typed (signature-family) fields."""
    return [f.name for f in fields(cls) if f.default_factory is dict]


def _from_dataset_with_children[T: DataclassInstance](
    cls: type[T],
    ds: xr.Dataset,
    children: dict[str, dict[str, xr.DataArray]] | None,
) -> T:
    """Rebuild a table dataclass from its Dataset node plus netCDF child groups.

    Plain DataArray fields come from *ds* variables; dict-typed fields
    (signature families) come from *children*, keyed by field name.
    """
    children = children or {}
    dict_names = set(_dict_field_names(cls))
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        kwargs[f.name] = children.get(f.name, {}) if f.name in dict_names else ds.get(f.name)
    return cls(**kwargs)


@dataclass
class _SizingArrays:
    min: xr.DataArray | None = None
    max: xr.DataArray | None = None
    mandatory: xr.DataArray | None = None
    effects_per_size: dict[str, xr.DataArray] = field(default_factory=dict)  # signature -> stacked rows
    effects_fixed: dict[str, xr.DataArray] = field(default_factory=dict)  # signature -> stacked rows

    def __post_init__(self) -> None:
        """Validate min >= 0 and max >= min."""
        if self.min is not None:
            mask = self.min < 0
            if mask.any():
                raise ValueError(f'Sizing.size_min < 0 on {list(self.min.coords[self.min.dims[0]][mask].values)}')
        if self.min is not None and self.max is not None:
            mask = self.max < self.min
            if mask.any():
                dim = self.min.dims[0]
                raise ValueError(f'Sizing.size_max < size_min on {list(self.min.coords[dim][mask].values)}')

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Sizing]],
        effect_ids: list[str],
        dim: str,
        entity_label: str = 'flow',
        period: pd.Index | None = None,
    ) -> Self:
        """Validate Sizing objects and collect into DataArrays.

        Args:
            items: Pairs of (element_id, Sizing).
            effect_ids: Known effect ids for validation.
            dim: Dimension name for the resulting arrays.
            entity_label: Entity coord name on stacked effect rows.
            period: Period index for period-varying effects.
        """
        if not items:
            return cls()

        effect_set = set(effect_ids)
        envelope_coords: dict[str, Any] = {'period': period} if period is not None else {}

        ids: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []
        mandatories: list[bool] = []

        for item_id, s in items:
            ids.append(item_id)
            mins.append(s.size_min)
            maxs.append(s.size_max)
            mandatories.append(s.mandatory)

        coords = {dim: ids}
        return cls(
            min=xr.DataArray(np.array(mins), dims=[dim], coords=coords),
            max=xr.DataArray(np.array(maxs), dims=[dim], coords=coords),
            mandatory=xr.DataArray(np.array(mandatories), dims=[dim], coords=coords),
            effects_per_size=_stack_element_effects(
                [(i, s.effects_per_size) for i, s in items],
                effect_set,
                envelope_coords,
                entity_label=entity_label,
                what='Sizing.effects_per_size',
            ),
            effects_fixed=_stack_element_effects(
                [(i, s.effects_fixed) for i, s in items],
                effect_set,
                envelope_coords,
                entity_label=entity_label,
                what='Sizing.effects_fixed',
            ),
        )


@dataclass
class _InvestmentArrays:
    min: xr.DataArray | None = None  # (invest_dim,)
    max: xr.DataArray | None = None  # (invest_dim,)
    mandatory: xr.DataArray | None = None  # (invest_dim,)
    lifetime: xr.DataArray | None = None  # (invest_dim,) — NaN = forever
    prior_size: xr.DataArray | None = None  # (invest_dim,)
    effects_per_size_at_build: dict[str, xr.DataArray] = field(default_factory=dict)  # once
    effects_fixed_at_build: dict[str, xr.DataArray] = field(default_factory=dict)  # once
    effects_per_size_recurring: dict[str, xr.DataArray] = field(default_factory=dict)
    effects_fixed_recurring: dict[str, xr.DataArray] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Investment]],
        effect_ids: list[str],
        dim: str,
        entity_label: str = 'flow',
        period: pd.Index | None = None,
    ) -> Self:
        """Validate Investment objects and collect into DataArrays.

        Args:
            items: Pairs of (element_id, Investment).
            effect_ids: Known effect ids for validation.
            dim: Dimension name for the resulting arrays.
            entity_label: Entity coord name on stacked effect rows.
            period: Period index for period-varying effects.
        """
        if not items:
            return cls()

        effect_set = set(effect_ids)
        envelope_coords: dict[str, Any] = {'period': period} if period is not None else {}

        ids: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []
        mandatories: list[bool] = []
        lifetimes: list[float] = []
        prior_sizes: list[float] = []

        for item_id, inv in items:
            if inv.size_max < inv.size_min:
                raise ValueError(f'Investment on {item_id!r}: size_max ({inv.size_max}) < size_min ({inv.size_min})')
            if inv.prior_size < 0:
                raise ValueError(f'Investment on {item_id!r}: prior_size must be >= 0, got {inv.prior_size}')
            if inv.lifetime is not None and inv.lifetime <= 0:
                raise ValueError(f'Investment on {item_id!r}: lifetime must be positive, got {inv.lifetime}')

            ids.append(item_id)
            mins.append(inv.size_min)
            maxs.append(inv.size_max)
            mandatories.append(inv.mandatory)
            lifetimes.append(float(inv.lifetime) if inv.lifetime is not None else np.nan)
            prior_sizes.append(inv.prior_size)

        def stack(what: str, dicts: list[tuple[str, dict[str, Variate]]]) -> dict[str, xr.DataArray]:
            return _stack_element_effects(dicts, effect_set, envelope_coords, entity_label=entity_label, what=what)

        coords = {dim: ids}
        return cls(
            min=xr.DataArray(np.array(mins), dims=[dim], coords=coords),
            max=xr.DataArray(np.array(maxs), dims=[dim], coords=coords),
            mandatory=xr.DataArray(np.array(mandatories), dims=[dim], coords=coords),
            lifetime=xr.DataArray(np.array(lifetimes), dims=[dim], coords=coords),
            prior_size=xr.DataArray(np.array(prior_sizes), dims=[dim], coords=coords),
            effects_per_size_at_build=stack(
                'Investment.effects_per_size_at_build', [(i, v.effects_per_size_at_build) for i, v in items]
            ),
            effects_fixed_at_build=stack(
                'Investment.effects_fixed_at_build', [(i, v.effects_fixed_at_build) for i, v in items]
            ),
            effects_per_size_recurring=stack(
                'Investment.effects_per_size_recurring', [(i, v.effects_per_size_recurring) for i, v in items]
            ),
            effects_fixed_recurring=stack(
                'Investment.effects_fixed_recurring', [(i, v.effects_fixed_recurring) for i, v in items]
            ),
        )


@dataclass
class _StatusArrays:
    uptime_min: xr.DataArray | None = None  # (dim,)
    uptime_max: xr.DataArray | None = None  # (dim,)
    downtime_min: xr.DataArray | None = None  # (dim,)
    downtime_max: xr.DataArray | None = None  # (dim,)
    initial: xr.DataArray | None = None  # (dim,) — NaN = free
    effects_running: dict[str, xr.DataArray] = field(default_factory=dict)  # signature -> stacked rows
    effects_startup: dict[str, xr.DataArray] = field(default_factory=dict)  # signature -> stacked rows
    previous_uptime: xr.DataArray | None = None  # (dim,) — hours, NaN = no prior
    previous_downtime: xr.DataArray | None = None  # (dim,) — hours, NaN = no prior
    governed_flows: xr.DataArray | None = None  # (dim, governed_idx) — only for component status

    def __post_init__(self) -> None:
        """Validate durations >= 0 and max >= min where both given."""
        for name in ('uptime_min', 'uptime_max', 'downtime_min', 'downtime_max'):
            arr: xr.DataArray | None = getattr(self, name)
            if arr is not None:
                mask = (~np.isnan(arr)) & (arr < 0)
                if mask.any():
                    dim = arr.dims[0]
                    raise ValueError(f'Status.{name} < 0 on {list(arr.coords[dim][mask].values)}')

        if self.uptime_min is not None and self.uptime_max is not None:
            both = ~np.isnan(self.uptime_min) & ~np.isnan(self.uptime_max)
            bad = both & (self.uptime_max < self.uptime_min)
            if bad.any():
                dim = self.uptime_min.dims[0]
                raise ValueError(f'Status.uptime_max < uptime_min on {list(self.uptime_min.coords[dim][bad].values)}')

        if self.downtime_min is not None and self.downtime_max is not None:
            both = ~np.isnan(self.downtime_min) & ~np.isnan(self.downtime_max)
            bad = both & (self.downtime_max < self.downtime_min)
            if bad.any():
                dim = self.downtime_min.dims[0]
                raise ValueError(
                    f'Status.downtime_max < downtime_min on {list(self.downtime_min.coords[dim][bad].values)}'
                )

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Status]],
        effect_ids: list[str],
        time: TimeIndex,
        dim: str,
        entity_label: str = 'flow',
        prior_rates_map: dict[str, list[float]] | None = None,
        dt: float = 1.0,
        period: pd.Index | None = None,
        governed_flows_map: dict[str, list[str]] | None = None,
    ) -> Self:
        """Validate Status objects and collect into DataArrays.

        Args:
            items: Pairs of (id, Status).
            effect_ids: Known effect ids for validation.
            time: Time index for effect arrays.
            dim: Dimension name for the resulting arrays.
            entity_label: Entity coord name on stacked effect rows —
                matches the governing binary variable's dim.
            prior_rates_map: Item id to prior flow rates (MW) before horizon.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for period-varying effects.
            governed_flows_map: Item id to ids of flows the status governs.
                Only populated for component-level status; emits a 2D
                ``(dim, governed_idx)`` string array.
        """
        from fluxopt.constraints.status import compute_previous_duration

        if not items:
            return cls()

        prior_rates_map = prior_rates_map or {}
        effect_set = set(effect_ids)
        envelope_coords: dict[str, Any] = {'time': time}
        if period is not None:
            envelope_coords['period'] = period

        ids: list[str] = []
        min_ups: list[float] = []
        max_ups: list[float] = []
        min_downs: list[float] = []
        max_downs: list[float] = []
        initials: list[float] = []
        prev_ups: list[float] = []
        prev_downs: list[float] = []

        for item_id, s in items:
            ids.append(item_id)
            min_ups.append(s.uptime_min if s.uptime_min is not None else np.nan)
            max_ups.append(s.uptime_max if s.uptime_max is not None else np.nan)
            min_downs.append(s.downtime_min if s.downtime_min is not None else np.nan)
            max_downs.append(s.downtime_max if s.downtime_max is not None else np.nan)

            prior = prior_rates_map.get(item_id)
            if prior is not None:
                initials.append(1.0 if prior[-1] > 0 else 0.0)
                prior_da = xr.DataArray(prior, dims=['_prior_t'])
                prev_ups.append(compute_previous_duration(prior_da, target_state=1, dt=dt))
                prev_downs.append(compute_previous_duration(prior_da, target_state=0, dt=dt))
            else:
                initials.append(np.nan)
                prev_ups.append(np.nan)
                prev_downs.append(np.nan)

        effects_running = _stack_element_effects(
            [(i, s.effects_per_running_hour) for i, s in items],
            effect_set,
            envelope_coords,
            entity_label=entity_label,
            what='Status.effects_per_running_hour',
        )
        effects_startup = _stack_element_effects(
            [(i, s.effects_per_startup) for i, s in items],
            effect_set,
            envelope_coords,
            entity_label=entity_label,
            what='Status.effects_per_startup',
        )

        coords = {dim: ids}

        prev_up_arr = np.array(prev_ups)
        prev_down_arr = np.array(prev_downs)

        governed: xr.DataArray | None = None
        if governed_flows_map:
            max_n = max(len(governed_flows_map.get(i, [])) for i in ids)
            if max_n > 0:
                rows = [
                    governed_flows_map.get(i, []) + [''] * (max_n - len(governed_flows_map.get(i, []))) for i in ids
                ]
                governed = xr.DataArray(
                    np.array(rows, dtype=object),
                    dims=[dim, 'governed_idx'],
                    coords={dim: ids},
                )

        return cls(
            uptime_min=xr.DataArray(np.array(min_ups), dims=[dim], coords=coords),
            uptime_max=xr.DataArray(np.array(max_ups), dims=[dim], coords=coords),
            downtime_min=xr.DataArray(np.array(min_downs), dims=[dim], coords=coords),
            downtime_max=xr.DataArray(np.array(max_downs), dims=[dim], coords=coords),
            initial=xr.DataArray(np.array(initials), dims=[dim], coords=coords),
            effects_running=effects_running,
            effects_startup=effects_startup,
            previous_uptime=xr.DataArray(prev_up_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_up_arr))
            else None,
            previous_downtime=xr.DataArray(prev_down_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_down_arr))
            else None,
            governed_flows=governed,
        )


@dataclass
class FlowsData:
    bound_type: xr.DataArray  # (flow,) — 'unsized' | 'bounded' | 'profile'
    rel_lb: xr.DataArray  # (flow, time[, period])
    rel_ub: xr.DataArray  # (flow, time[, period])
    fixed_profile: xr.DataArray  # (flow, time[, period]) — NaN where not fixed
    size: xr.DataArray  # (flow,) — NaN for unsized
    # signature -> (contribution, *signature_dims); see _stack_contributions; {} = no effects
    effect_coeff: dict[str, xr.DataArray] = field(default_factory=dict)
    flow_hours_min: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    flow_hours_max: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    load_factor_min: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    load_factor_max: xr.DataArray | None = None  # (flow,) — NaN = unbounded, per period
    ramp_up: xr.DataArray | None = None  # (flow, time[, period]) — NaN = no limit [1/h]
    ramp_down: xr.DataArray | None = None  # (flow, time[, period]) — NaN = no limit [1/h]
    sizing_min: xr.DataArray | None = None  # (sizing_flow,)
    sizing_max: xr.DataArray | None = None  # (sizing_flow,)
    sizing_mandatory: xr.DataArray | None = None  # (sizing_flow,)
    sizing_effects_per_size: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'flow'
    sizing_effects_fixed: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'flow'
    status_uptime_min: xr.DataArray | None = None  # (status_flow,)
    status_uptime_max: xr.DataArray | None = None  # (status_flow,)
    status_downtime_min: xr.DataArray | None = None  # (status_flow,)
    status_downtime_max: xr.DataArray | None = None  # (status_flow,)
    status_initial: xr.DataArray | None = None  # (status_flow,)
    status_effects_running: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'flow'
    status_effects_startup: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'flow'
    status_previous_uptime: xr.DataArray | None = None  # (status_flow,)
    status_previous_downtime: xr.DataArray | None = None  # (status_flow,)
    invest_min: xr.DataArray | None = None  # (invest_flow,)
    invest_max: xr.DataArray | None = None  # (invest_flow,)
    invest_mandatory: xr.DataArray | None = None  # (invest_flow,)
    invest_lifetime: xr.DataArray | None = None  # (invest_flow,) — NaN = forever
    invest_prior_size: xr.DataArray | None = None  # (invest_flow,)
    invest_effects_per_size_at_build: dict[str, xr.DataArray] = field(default_factory=dict)  # once, coord 'flow'
    invest_effects_fixed_at_build: dict[str, xr.DataArray] = field(default_factory=dict)  # once, coord 'flow'
    invest_effects_per_size_recurring: dict[str, xr.DataArray] = field(default_factory=dict)  # coord 'flow'
    invest_effects_fixed_recurring: dict[str, xr.DataArray] = field(default_factory=dict)  # coord 'flow'
    cstatus_uptime_min: xr.DataArray | None = None  # (cstatus_component,)
    cstatus_uptime_max: xr.DataArray | None = None  # (cstatus_component,)
    cstatus_downtime_min: xr.DataArray | None = None  # (cstatus_component,)
    cstatus_downtime_max: xr.DataArray | None = None  # (cstatus_component,)
    cstatus_initial: xr.DataArray | None = None  # (cstatus_component,) — NaN = free
    cstatus_effects_running: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'component'
    cstatus_effects_startup: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'component'
    cstatus_previous_uptime: xr.DataArray | None = None  # (cstatus_component,)
    cstatus_previous_downtime: xr.DataArray | None = None  # (cstatus_component,)
    cstatus_governed_flows: xr.DataArray | None = None  # (cstatus_component, governed_idx) — qualified flow ids

    def __post_init__(self) -> None:
        """Validate relative bounds (non-negative, lb <= ub) and contribution pair uniqueness."""
        reduce_dims = [d for d in self.rel_lb.dims if d != 'flow']
        bad_neg = (self.rel_lb < -1e-12).any(reduce_dims)
        if bad_neg.any():
            raise ValueError(f'Negative lower bounds on flows: {list(self.rel_lb.coords["flow"][bad_neg].values)}')
        bad_order = (self.rel_lb > self.rel_ub + 1e-12).any(reduce_dims)
        if bad_order.any():
            raise ValueError(
                f'Lower bound > upper bound on flows: {list(self.rel_lb.coords["flow"][bad_order].values)}'
            )
        for family, entity_label in (
            ('effect_coeff', 'flow'),
            ('sizing_effects_per_size', 'flow'),
            ('sizing_effects_fixed', 'flow'),
            ('status_effects_running', 'flow'),
            ('status_effects_startup', 'flow'),
            ('invest_effects_per_size_at_build', 'flow'),
            ('invest_effects_fixed_at_build', 'flow'),
            ('invest_effects_per_size_recurring', 'flow'),
            ('invest_effects_fixed_recurring', 'flow'),
            ('cstatus_effects_running', 'component'),
            ('cstatus_effects_startup', 'component'),
        ):
            _validate_stacked_pairs(family, getattr(self, family), entity_label)

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, children: dict[str, dict[str, xr.DataArray]] | None = None) -> Self:
        """Deserialize from xr.Dataset plus netCDF child groups.

        Args:
            ds: Dataset with matching variable names.
            children: Signature-grouped families parsed from the netCDF
                child groups, keyed by field name (see ``ModelData.from_netcdf``).
        """
        return _from_dataset_with_children(cls, ds, children)

    @classmethod
    def build(
        cls,
        flows: list[Flow],
        time: TimeIndex,
        effects: list[Effect],
        dt: float = 1.0,
        period: pd.Index | None = None,
        component_status_items: list[tuple[str, Status, list[str]]] | None = None,
    ) -> Self:
        """Build FlowsData from element objects.

        Args:
            flows: All collected flows with qualified ids.
            time: Time index.
            effects: Effect definitions for cost coefficients.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for multi-period models. When provided,
                ``effect_coeff``, ``rel_lb``, ``rel_ub`` and ``fixed_profile``
                gain a ``period`` dimension so that ``effects_per_flow_hour``,
                ``relative_rate_min``, ``relative_rate_max`` and
                ``fixed_relative_profile`` can vary across periods.
            component_status_items: Component-level status entries as
                ``(component_id, Status, [governed flow ids])``. Each entry
                produces an on/startup/shutdown binary keyed by the
                component, gating all listed flows together.
        """
        from fluxopt.elements import Investment, Sizing

        flow_ids = [f.id for f in flows]
        effect_ids = [e.id for e in effects]
        effect_set = set(effect_ids)

        bound_type: list[str] = []
        rel_lbs: list[xr.DataArray] = []
        rel_ubs: list[xr.DataArray] = []
        profiles: list[xr.DataArray] = []
        size_vals = np.full(len(flows), np.nan)
        fh_min_vals = np.full(len(flows), np.nan)
        fh_max_vals = np.full(len(flows), np.nan)
        lf_min_vals = np.full(len(flows), np.nan)
        lf_max_vals = np.full(len(flows), np.nan)
        ramp_ups: list[xr.DataArray] = []
        ramp_downs: list[xr.DataArray] = []
        has_ramp_up = False
        has_ramp_down = False
        contrib_flows: list[str] = []
        contrib_effects: list[str] = []
        contrib_vals: list[xr.DataArray] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []
        status_items: list[tuple[str, Status]] = []
        prior_rates_map: dict[str, list[float]] = {}

        envelope_coords: dict[str, Any] = {'time': time}
        if period is not None:
            envelope_coords['period'] = period
        nan_envelope = xr.DataArray(
            np.full([len(v) for v in envelope_coords.values()], np.nan),
            dims=list(envelope_coords),
            coords=envelope_coords,
        )

        for i, f in enumerate(flows):
            rel_lbs.append(as_dataarray(f.relative_rate_min, envelope_coords))
            rel_ubs.append(as_dataarray(f.relative_rate_max, envelope_coords))

            if isinstance(f.size, Sizing):
                sizing_items.append((f.id, f.size))
            elif isinstance(f.size, Investment):
                invest_items.append((f.id, f.size))
            elif f.size is not None:
                size_vals[i] = f.size

            if f.flow_hours_min is not None:
                fh_min_vals[i] = f.flow_hours_min
            if f.flow_hours_max is not None:
                fh_max_vals[i] = f.flow_hours_max
            if f.load_factor_min is not None:
                lf_min_vals[i] = f.load_factor_min
            if f.load_factor_max is not None:
                lf_max_vals[i] = f.load_factor_max

            has_ramp_up = has_ramp_up or f.ramp_up_per_hour is not None
            has_ramp_down = has_ramp_down or f.ramp_down_per_hour is not None
            ramp_ups.append(
                as_dataarray(f.ramp_up_per_hour, envelope_coords) if f.ramp_up_per_hour is not None else nan_envelope
            )
            ramp_downs.append(
                as_dataarray(f.ramp_down_per_hour, envelope_coords)
                if f.ramp_down_per_hour is not None
                else nan_envelope
            )

            if f.fixed_relative_profile is not None:
                profiles.append(as_dataarray(f.fixed_relative_profile, envelope_coords))
                bound_type.append('profile')
            elif f.size is None:
                profiles.append(nan_envelope)
                bound_type.append('unsized')
            else:
                profiles.append(nan_envelope)
                bound_type.append('bounded')

            for effect_label, factor in f.effects_per_flow_hour.items():
                if effect_label not in effect_set:
                    raise ValueError(f'Unknown effect {effect_label!r} in Flow.effects_per_flow_hour on {f.id!r}')
                contrib_flows.append(f.id)
                contrib_effects.append(effect_label)
                contrib_vals.append(as_dataarray(factor, envelope_coords, broadcast=False))

            if f.status is not None:
                status_items.append((f.id, f.status))

            if f.prior_rates is not None:
                prior_rates_map[f.id] = f.prior_rates

        flow_idx = pd.Index(flow_ids, name='flow')
        effect_coeff = _stack_contributions(contrib_flows, contrib_effects, contrib_vals, envelope_coords)
        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_flow', entity_label='flow', period=period)
        inv = _InvestmentArrays.build(invest_items, effect_ids, dim='invest_flow', entity_label='flow', period=period)
        st = _StatusArrays.build(
            status_items, effect_ids, time, dim='status_flow', prior_rates_map=prior_rates_map, dt=dt, period=period
        )

        cst = _StatusArrays.build(
            [(cid, s) for cid, s, _ in (component_status_items or [])],
            effect_ids,
            time,
            dim='cstatus_component',
            entity_label='component',
            period=period,
            governed_flows_map={cid: gov for cid, _, gov in (component_status_items or [])} or None,
        )

        return cls(
            bound_type=xr.DataArray(bound_type, dims=['flow'], coords={'flow': flow_ids}),
            rel_lb=fast_concat(rel_lbs, flow_idx),
            rel_ub=fast_concat(rel_ubs, flow_idx),
            fixed_profile=fast_concat(profiles, flow_idx),
            size=xr.DataArray(size_vals, dims=['flow'], coords={'flow': flow_ids}),
            effect_coeff=effect_coeff,
            flow_hours_min=_flow_bound_or_none(fh_min_vals, flow_ids),
            flow_hours_max=_flow_bound_or_none(fh_max_vals, flow_ids),
            load_factor_min=_flow_bound_or_none(lf_min_vals, flow_ids),
            load_factor_max=_flow_bound_or_none(lf_max_vals, flow_ids),
            ramp_up=fast_concat(ramp_ups, flow_idx) if has_ramp_up else None,
            ramp_down=fast_concat(ramp_downs, flow_idx) if has_ramp_down else None,
            sizing_min=sz.min,
            sizing_max=sz.max,
            sizing_mandatory=sz.mandatory,
            sizing_effects_per_size=sz.effects_per_size,
            sizing_effects_fixed=sz.effects_fixed,
            status_uptime_min=st.uptime_min,
            status_uptime_max=st.uptime_max,
            status_downtime_min=st.downtime_min,
            status_downtime_max=st.downtime_max,
            status_initial=st.initial,
            status_effects_running=st.effects_running,
            status_effects_startup=st.effects_startup,
            status_previous_uptime=st.previous_uptime,
            status_previous_downtime=st.previous_downtime,
            invest_min=inv.min,
            invest_max=inv.max,
            invest_mandatory=inv.mandatory,
            invest_lifetime=inv.lifetime,
            invest_prior_size=inv.prior_size,
            invest_effects_per_size_at_build=inv.effects_per_size_at_build,
            invest_effects_fixed_at_build=inv.effects_fixed_at_build,
            invest_effects_per_size_recurring=inv.effects_per_size_recurring,
            invest_effects_fixed_recurring=inv.effects_fixed_recurring,
            cstatus_uptime_min=cst.uptime_min,
            cstatus_uptime_max=cst.uptime_max,
            cstatus_downtime_min=cst.downtime_min,
            cstatus_downtime_max=cst.downtime_max,
            cstatus_initial=cst.initial,
            cstatus_effects_running=cst.effects_running,
            cstatus_effects_startup=cst.effects_startup,
            cstatus_previous_uptime=cst.previous_uptime,
            cstatus_previous_downtime=cst.previous_downtime,
            cstatus_governed_flows=cst.governed_flows,
        )


_SIGNATURE_DIM_ORDER = ('time', 'period')  # canonical envelope dim order for signature names


def _signature_name(dims: tuple[str, ...]) -> str:
    """Name a dims signature: ``()`` -> 'scalar', ``('time', 'period')`` -> 'time_period'."""
    return '_'.join(dims) if dims else 'scalar'


def _stack_contributions(
    entities: list[str],
    effects: list[str],
    values: list[xr.DataArray],
    envelope_coords: dict[str, Any],
    *,
    entity_label: str = 'flow',
) -> dict[str, xr.DataArray]:
    """Group per-pair effect coefficients by dims signature and stack each group.

    One row per (entity, effect) pair that has coefficients — pairs without
    coefficients are simply absent, never stored as zeros. Rows are grouped
    by the *natural* dims of their input (``()``, ``('time',)``,
    ``('period',)``, ``('time', 'period')``) so scalar coefficients never
    carry a time envelope; groups without rows are absent from the dict.

    Within each group, ``contribution`` is a bare positional dim (no index
    coord, so it can never participate in alignment); each row's pair is
    labeled by the non-dim coords *entity_label* / ``effect``. Signature
    arrays never merge into one Dataset — neither at runtime nor on disk
    (each is its own self-contained netCDF group) — so the plain coord
    names cannot collide with the real dims.

    Args:
        entities: Entity id per contribution row (flow / component / storage).
        effects: Effect id per contribution row.
        values: Natural-dims coefficient per row
            (``as_dataarray(..., broadcast=False)``).
        envelope_coords: Operational coords (``time``[, ``period``]) the
            signature dims draw from.
        entity_label: Coord name for the entity labels — matches the dim of
            the solver variable the channel multiplies (``flow``,
            ``component``, ``storage``).

    Returns:
        Mapping of signature name -> stacked coefficients
        ``(contribution, *signature_dims)``; empty when there are no rows.
    """
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, v in enumerate(values):
        sig = tuple(d for d in _SIGNATURE_DIM_ORDER if d in v.dims)
        groups.setdefault(sig, []).append(i)

    out: dict[str, xr.DataArray] = {}
    for sig, idxs in groups.items():
        out[_signature_name(sig)] = xr.DataArray(
            np.stack([values[i].transpose(*sig).values for i in idxs]),
            dims=['contribution', *sig],
            coords={
                **{d: envelope_coords[d] for d in sig},
                entity_label: ('contribution', [entities[i] for i in idxs]),
                'effect': ('contribution', [effects[i] for i in idxs]),
            },
        )
    return out


def _stack_element_effects(
    items: list[tuple[str, dict[str, Variate]]],
    effect_set: set[str],
    envelope_coords: dict[str, Any],
    *,
    entity_label: str,
    what: str,
) -> dict[str, xr.DataArray]:
    """Collect per-element effect dicts into a signature-grouped family.

    Args:
        items: Pairs of (element id, ``{effect: factor}``).
        effect_set: Known effect ids for validation.
        envelope_coords: Operational coords the factors may span.
        entity_label: Coord name for entity labels (see
            :func:`_stack_contributions`).
        what: Parameter name for error messages
            (e.g. ``'Status.effects_per_startup'``).
    """
    entities: list[str] = []
    effects: list[str] = []
    values: list[xr.DataArray] = []
    for item_id, factors in items:
        for ek, ev in factors.items():
            if ek not in effect_set:
                raise ValueError(f'Unknown effect {ek!r} in {what} on {item_id!r}')
            entities.append(item_id)
            effects.append(ek)
            values.append(as_dataarray(ev, envelope_coords, broadcast=False))
    return _stack_contributions(entities, effects, values, envelope_coords, entity_label=entity_label)


def _split_rows(
    coeffs: dict[str, xr.DataArray],
    entity_dim: str,
    is_mandatory: dict[str, bool],
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
    """Partition signature-grouped rows by a per-entity boolean.

    Args:
        coeffs: Signature-grouped stacked coefficients.
        entity_dim: Entity label coord on the contribution dim.
        is_mandatory: Entity id -> True when the entity is mandatory.

    Returns:
        ``(optional_rows, mandatory_rows)`` with empty signatures dropped.
    """
    optional: dict[str, xr.DataArray] = {}
    mandatory: dict[str, xr.DataArray] = {}
    for sig, arr in coeffs.items():
        mask = np.array([bool(is_mandatory[str(e)]) for e in arr.coords[entity_dim].values])
        if (~mask).any():
            optional[sig] = arr.isel(contribution=np.nonzero(~mask)[0])
        if mask.any():
            mandatory[sig] = arr.isel(contribution=np.nonzero(mask)[0])
    return optional, mandatory


def _validate_stacked_pairs(family: str, coeffs: dict[str, xr.DataArray], entity_label: str) -> None:
    """Raise when an (entity, effect) pair appears twice across a family's signatures.

    Args:
        family: Field name for the error message.
        coeffs: Signature-grouped stacked coefficients.
        entity_label: Entity coord name on the contribution dim.
    """
    pairs: list[tuple[str, str]] = []
    for arr in coeffs.values():
        pairs += list(zip(arr.coords[entity_label].values, arr.coords['effect'].values, strict=True))
    if len(pairs) != len(set(pairs)):
        dupes = sorted({p for p in pairs if pairs.count(p) > 1})
        raise ValueError(f'Duplicate ({entity_label}, effect) contribution pairs in {family}: {dupes}')


def _flow_bound_or_none(vals: np.ndarray, flow_ids: list[str]) -> xr.DataArray | None:
    """Wrap per-flow bound values as a (flow,) DataArray, or None if all NaN.

    Args:
        vals: Bound value per flow; NaN = unbounded.
        flow_ids: Flow coordinate labels.
    """
    if np.all(np.isnan(vals)):
        return None
    return xr.DataArray(vals, dims=['flow'], coords={'flow': flow_ids})


def _carrier_dim_id(flow: Flow) -> str:
    """Return the carrier dimension coordinate value for a flow.

    Single-node carriers use the carrier id directly.
    Multi-node carriers use ``carrier_id:node``.

    Args:
        flow: Flow with carrier (and optional node).
    """
    from fluxopt.elements import node_id

    if flow.node is not None:
        return node_id(flow.carrier, flow.node)
    return flow.carrier


@dataclass
class CarriersData:
    flow_coeff: xr.DataArray  # (carrier, flow) — +1/-1/NaN
    unit: xr.DataArray  # (carrier,) — energy unit label
    color: xr.DataArray  # (carrier,) — plot color ('' if unset)
    description: xr.DataArray  # (carrier,) — human-readable description

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with ``flow_coeff``, ``unit``, ``color``, ``description``.
        """
        return cls(
            flow_coeff=ds['flow_coeff'],
            unit=ds['unit'],
            color=ds['color'],
            description=ds['description'],
        )

    @classmethod
    def build(cls, carriers: list[Carrier], flows: list[Flow], carrier_coeff: dict[str, float]) -> Self:
        """Build CarriersData from explicit carrier declarations.

        Args:
            carriers: Declared carriers.
            flows: All collected flows.
            carrier_coeff: Mapping of flow id to +1 (produces) or -1 (consumes).
        """
        from fluxopt.elements import node_id

        flow_ids = [f.id for f in flows]
        # Build carrier dim ids from explicit declarations
        carrier_ids: list[str] = []
        for c in carriers:
            if c.nodes:
                carrier_ids.extend(node_id(c.id, node) for node in c.nodes)
            else:
                carrier_ids.append(c.id)

        coeff = np.full((len(carrier_ids), len(flow_ids)), np.nan)
        for f in flows:
            ci = carrier_ids.index(_carrier_dim_id(f))
            fi = flow_ids.index(f.id)
            coeff[ci, fi] = carrier_coeff[f.id]

        # Expand carrier metadata to match carrier dim (one entry per node)
        units: list[str] = []
        colors: list[str] = []
        descriptions: list[str] = []
        for c in carriers:
            n = max(len(c.nodes), 1)
            units.extend([c.unit] * n)
            colors.extend([c.color or ''] * n)
            descriptions.extend([c.description] * n)

        return cls(
            flow_coeff=xr.DataArray(coeff, dims=['carrier', 'flow'], coords={'carrier': carrier_ids, 'flow': flow_ids}),
            unit=xr.DataArray(units, dims=['carrier'], coords={'carrier': carrier_ids}),
            color=xr.DataArray(colors, dims=['carrier'], coords={'carrier': carrier_ids}),
            description=xr.DataArray(descriptions, dims=['carrier'], coords={'carrier': carrier_ids}),
        )


@dataclass
class ConvertersData:
    pair_coeff: xr.DataArray  # (pair, eq_idx, time) — non-zero coefficients only
    pair_converter: xr.DataArray  # (pair,) — converter id per pair
    pair_flow: xr.DataArray  # (pair,) — flow id per pair
    eq_mask: xr.DataArray  # (converter, eq_idx)

    @property
    def flow_coeff(self) -> xr.DataArray:
        """Dense (converter, eq_idx, flow, time) view for inspection."""
        conv_ids = list(dict.fromkeys(self.pair_converter.values))
        flow_ids = list(dict.fromkeys(self.pair_flow.values))
        eq_idx = list(self.pair_coeff.coords['eq_idx'].values)
        time = self.pair_coeff.coords['time']

        dense = xr.DataArray(
            np.full((len(conv_ids), len(eq_idx), len(flow_ids), len(time)), np.nan),
            dims=['converter', 'eq_idx', 'flow', 'time'],
            coords={'converter': conv_ids, 'eq_idx': eq_idx, 'flow': flow_ids, 'time': time},
        )
        for i in range(len(self.pair_converter)):
            conv_id = str(self.pair_converter.values[i])
            flow_id = str(self.pair_flow.values[i])
            dense.loc[conv_id, :, flow_id, :] = self.pair_coeff.isel(pair=i)
        return dense

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with pair-based converter coefficient variables.
        """
        return cls(
            pair_coeff=ds['pair_coeff'],
            pair_converter=ds['pair_converter'],
            pair_flow=ds['pair_flow'],
            eq_mask=ds['eq_mask'],
        )

    @classmethod
    def build(cls, converters: list[Converter], time: TimeIndex) -> Self | None:
        """Build ConvertersData with sparse pair-based conversion coefficients.

        Only linear converters are included; piecewise converters
        (``conversion is not None``) live in :class:`PiecewiseData`.

        Args:
            converters: Converter definitions.
            time: Time index.
        """
        converters = [c for c in converters if c.conversion is None]
        if not converters:
            return None

        conv_ids = [c.id for c in converters]
        max_eq = max(len(c.conversion_factors) for c in converters)
        n_time = len(time)
        eq_idx_list = list(range(max_eq))

        eq_mask_rows: list[np.ndarray] = []
        pairs_conv: list[str] = []
        pairs_flow: list[str] = []
        coeff_arrays: list[np.ndarray] = []

        for conv in converters:
            mask_row = np.zeros(max_eq, dtype=bool)
            for eq_i in range(len(conv.conversion_factors)):
                mask_row[eq_i] = True
            eq_mask_rows.append(mask_row)

            qid_to_short = {v: k for k, v in conv._short_to_id.items()}
            for flow in (*conv.inputs, *conv.outputs):
                short = qid_to_short[flow.id]
                eq_coeffs = np.zeros((max_eq, n_time))
                for eq_i, equation in enumerate(conv.conversion_factors):
                    if short in equation:
                        eq_coeffs[eq_i] = as_dataarray(equation[short], {'time': time}).values
                pairs_conv.append(conv.id)
                pairs_flow.append(flow.id)
                coeff_arrays.append(eq_coeffs)

        return cls(
            pair_coeff=xr.DataArray(
                np.array(coeff_arrays),
                dims=['pair', 'eq_idx', 'time'],
                coords={'eq_idx': eq_idx_list, 'time': time},
            ),
            pair_converter=xr.DataArray(pairs_conv, dims=['pair']),
            pair_flow=xr.DataArray(pairs_flow, dims=['pair']),
            eq_mask=xr.DataArray(
                np.array(eq_mask_rows),
                dims=['converter', 'eq_idx'],
                coords={'converter': conv_ids, 'eq_idx': eq_idx_list},
            ),
        )


@dataclass
class PiecewiseData:
    """Piecewise-linear conversion data for converters with ``PiecewiseConversion``.

    Stored sparsely as one row per (converter, flow) pair; the ``method``
    and ``availability`` arrays index by ``pw_converter``.
    """

    breakpoints: xr.DataArray  # (pw_pair, breakpoint, time)
    pair_converter: xr.DataArray  # (pw_pair,) — converter id
    pair_flow: xr.DataArray  # (pw_pair,) — qualified flow id
    pair_bound: xr.DataArray  # (pw_pair,) — '==' / '<=' / '>='
    method: xr.DataArray  # (pw_converter,) — 'auto' / 'sos2' / 'incremental' / 'lp'
    availability: xr.DataArray  # (pw_converter, time)
    has_status: xr.DataArray  # (pw_converter,) — bool

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with piecewise variables.
        """
        return cls(
            breakpoints=ds['breakpoints'],
            pair_converter=ds['pair_converter'],
            pair_flow=ds['pair_flow'],
            pair_bound=ds['pair_bound'],
            method=ds['method'],
            availability=ds['availability'],
            has_status=ds['has_status'],
        )

    def converter_ids(self) -> list[str]:
        """Return list of piecewise converter ids in original order."""
        return list(self.method.coords['pw_converter'].values)

    @classmethod
    def build(cls, converters: list[Converter], time: TimeIndex) -> Self | None:
        """Build PiecewiseData from converters with ``PiecewiseConversion``.

        Args:
            converters: Converter definitions; only those with
                ``conversion is not None`` are processed.
            time: Time index for breakpoint and availability arrays.
        """
        converters = [c for c in converters if c.conversion is not None]
        if not converters:
            return None

        conv_ids: list[str] = []
        methods: list[str] = []
        avail_slices: list[xr.DataArray] = []
        has_statuses: list[bool] = []

        pair_conv_ids: list[str] = []
        pair_flow_ids: list[str] = []
        pair_bounds: list[str] = []
        bp_slices: list[xr.DataArray] = []

        for conv in converters:
            assert conv.conversion is not None
            curve = conv.conversion
            conv_ids.append(conv.id)
            methods.append(curve.method)
            avail_slices.append(as_dataarray(curve.availability, {'time': time}))
            has_statuses.append(curve.status is not None)

            for short, pts, bound in curve._iter_normalized():
                qid = conv._short_to_id[short]
                bp_arrays = [as_dataarray(bp, {'time': time}) for bp in pts]
                bp_idx = pd.Index(range(len(bp_arrays)), name='breakpoint')
                bp_da = fast_concat(bp_arrays, bp_idx)
                pair_conv_ids.append(conv.id)
                pair_flow_ids.append(qid)
                pair_bounds.append(bound)
                bp_slices.append(bp_da)

        pair_idx = pd.Index(range(len(bp_slices)), name='pw_pair')
        breakpoints_da = fast_concat(bp_slices, pair_idx)

        conv_idx = pd.Index(conv_ids, name='pw_converter')
        availability = fast_concat(avail_slices, conv_idx)

        data = cls(
            breakpoints=breakpoints_da,
            pair_converter=xr.DataArray(pair_conv_ids, dims=['pw_pair']),
            pair_flow=xr.DataArray(pair_flow_ids, dims=['pw_pair']),
            pair_bound=xr.DataArray(pair_bounds, dims=['pw_pair']),
            method=xr.DataArray(methods, dims=['pw_converter'], coords={'pw_converter': conv_ids}),
            availability=availability,
            has_status=xr.DataArray(has_statuses, dims=['pw_converter'], coords={'pw_converter': conv_ids}),
        )
        data._warn_redundant_status()
        return data

    def _warn_redundant_status(self) -> None:
        """Warn for converters where Status is set but the curve includes a
        (0, ..., 0) breakpoint at any (breakpoint, timestep) position.

        When that's the case, the optimizer can sit at zero with ``active=1``,
        so the on/off binary is decoupled from the actual operating state and
        Status features will not behave as expected.
        """
        atol = 1e-9
        is_zero = abs(self.breakpoints) <= atol  # (pw_pair, breakpoint, time)
        for conv_id in self.converter_ids():
            if not bool(self.has_status.sel(pw_converter=conv_id).item()):
                continue
            mask = self.pair_converter.values == conv_id
            all_flows_zero = is_zero.isel(pw_pair=mask).all('pw_pair')  # (breakpoint, time)
            if bool(all_flows_zero.any().item()):
                warnings.warn(
                    f'PiecewiseConversion on converter {conv_id!r} has Status, '
                    'but the curve includes a (0, ..., 0) breakpoint. The '
                    'optimizer can sit at zero with status=on, decoupling the '
                    'binary from the actual operating state — Status features '
                    'will not behave as expected. If you want Status to work '
                    'as expected, drop the zero breakpoint so the only way to '
                    'produce zero is status=off.',
                    UserWarning,
                    stacklevel=4,
                )


def _detect_contribution_cycle(adjacency: dict[str, list[str]]) -> list[str] | None:
    """Return first cycle found in directed graph, or None.

    Args:
        adjacency: Mapping of node to list of neighbors (outgoing edges).
    """
    unvisited, in_stack, done = 0, 1, 2
    state: dict[str, int] = dict.fromkeys(adjacency, unvisited)
    path: list[str] = []

    def dfs(node: str) -> list[str] | None:
        state[node] = in_stack
        path.append(node)
        for neighbor in adjacency.get(node, []):
            if state[neighbor] == in_stack:
                i = path.index(neighbor)
                return [*path[i:], neighbor]
            if state[neighbor] == unvisited:
                result = dfs(neighbor)
                if result is not None:
                    return result
        path.pop()
        state[node] = done
        return None

    for node in adjacency:
        if state[node] == unvisited:
            cycle = dfs(node)
            if cycle is not None:
                return cycle
    return None


@dataclass
class EffectsData:
    total_min: xr.DataArray  # (effect,) — weighted total bound
    total_max: xr.DataArray  # (effect,) — weighted total bound
    periodic_min: xr.DataArray  # (effect[, period]) — per-period bound
    periodic_max: xr.DataArray  # (effect[, period]) — per-period bound
    cf_temporal: xr.DataArray | None = None  # (effect, source_effect, time, period?)
    period_weights: xr.DataArray | None = None  # (effect, period)

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with effect variables and attrs.
        """
        kwargs: dict[str, object] = {}
        for f in fields(cls):
            if f.name in ds.data_vars:
                kwargs[f.name] = ds[f.name]
            elif f.name in ds.attrs:
                kwargs[f.name] = ds.attrs[f.name]
            # else: rely on dataclass default (e.g. None for optional fields)
        return cls(**kwargs)  # pyrefly: ignore[bad-argument-type]

    @classmethod
    def build(
        cls,
        effects: list[Effect],
        time: TimeIndex,
        period: pd.Index | None = None,
    ) -> Self:
        """Build EffectsData from element objects.

        Args:
            effects: Effect definitions.
            time: Time index.
            period: Period index (multi-period only).
        """
        effect_ids = [e.id for e in effects]
        effect_set = set(effect_ids)
        n = len(effects)
        total_min = np.full(n, np.nan)
        total_max = np.full(n, np.nan)
        periodic_mins: list[xr.DataArray] = []
        periodic_maxs: list[xr.DataArray] = []

        # Periodic bounds are scalar in single-period models, (period,) in multi-period
        period_coords: dict[str, Any] = {'period': period} if period is not None else {}
        nan_periodic = (
            xr.DataArray(np.full(len(period), np.nan), dims=['period'], coords={'period': period})
            if period is not None
            else xr.DataArray(np.nan)
        )

        has_contributions = False
        for i, e in enumerate(effects):
            if e.total_min is not None:
                total_min[i] = e.total_min
            if e.total_max is not None:
                total_max[i] = e.total_max
            periodic_mins.append(
                as_dataarray(e.periodic_min, period_coords) if e.periodic_min is not None else nan_periodic
            )
            periodic_maxs.append(
                as_dataarray(e.periodic_max, period_coords) if e.periodic_max is not None else nan_periodic
            )
            if e.contribution_from:
                has_contributions = True

        # Build cross-effect contribution arrays
        cf_temporal: xr.DataArray | None = None
        if has_contributions:
            # Self-reference check
            for e in effects:
                for src_id in e.contribution_from:
                    if src_id == e.id:
                        raise ValueError(f'Effect {e.id!r} cannot reference itself in contribution_from')

            # Cycle check
            adjacency: dict[str, list[str]] = {eid: [] for eid in effect_ids}
            for e in effects:
                for src_id in e.contribution_from:
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in contribution_from on {e.id!r}')
                    adjacency[e.id].append(src_id)
            cycle = _detect_contribution_cycle(adjacency)
            if cycle is not None:
                raise ValueError(f'Circular contribution_from dependency: {" -> ".join(cycle)}')

            tmpl_t = _effect_template({'effect': effect_ids, 'source_effect': effect_ids, 'time': time}, period)
            temporal_mat = tmpl_t.zeros()
            for e in effects:
                for src_id, factor in e.contribution_from.items():
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in Effect.contribution_from on {e.id!r}')
                    temporal_mat.loc[e.id, src_id] = as_dataarray(factor, tmpl_t.as_da_coords)
            cf_temporal = temporal_mat

        effect_idx = pd.Index(effect_ids, name='effect')

        # Per-effect period weights
        pw: xr.DataArray | None = None
        if period is not None:
            has_pw = any(e.period_weights is not None for e in effects)
            n_periods = len(period)
            if has_pw:
                mat = np.full((n, n_periods), np.nan)
                for i, e in enumerate(effects):
                    if e.period_weights is not None:
                        if len(e.period_weights) != n_periods:
                            msg = f'Effect {e.id!r}: period_weights has {len(e.period_weights)} entries, expected {n_periods}'
                            raise ValueError(msg)
                        vals = np.asarray(e.period_weights, dtype=float)
                        if not np.all(np.isfinite(vals)) or not np.all(vals > 0):
                            msg = f'Effect {e.id!r}: period_weights must be positive and finite, got {vals}'
                            raise ValueError(msg)
                        mat[i] = vals
                pw = xr.DataArray(mat, dims=['effect', 'period'], coords={'effect': effect_ids, 'period': period})

        return cls(
            total_min=xr.DataArray(total_min, dims=['effect'], coords={'effect': effect_ids}),
            total_max=xr.DataArray(total_max, dims=['effect'], coords={'effect': effect_ids}),
            periodic_min=fast_concat(periodic_mins, effect_idx),
            periodic_max=fast_concat(periodic_maxs, effect_idx),
            cf_temporal=cf_temporal,
            period_weights=pw,
        )


@dataclass
class StoragesData:
    capacity: xr.DataArray  # (storage,)
    eta_c: xr.DataArray  # (storage, time)
    eta_d: xr.DataArray  # (storage, time)
    loss: xr.DataArray  # (storage, time)
    rel_level_lb: xr.DataArray  # (storage, time)
    rel_level_ub: xr.DataArray  # (storage, time)
    prior_level: xr.DataArray  # (storage,) — NaN if not set
    cyclic: xr.DataArray  # (storage,)
    charge_flow: xr.DataArray  # (storage,) — str
    discharge_flow: xr.DataArray  # (storage,) — str
    final_level_min: xr.DataArray | None = None  # (storage,) — NaN = unbounded [MWh]
    final_level_max: xr.DataArray | None = None  # (storage,) — NaN = unbounded [MWh]
    prevent_simultaneous: xr.DataArray | None = None  # (storage,) — bool
    sizing_min: xr.DataArray | None = None  # (sizing_storage,)
    sizing_max: xr.DataArray | None = None  # (sizing_storage,)
    sizing_mandatory: xr.DataArray | None = None  # (sizing_storage,)
    sizing_effects_per_size: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'storage'
    sizing_effects_fixed: dict[str, xr.DataArray] = field(default_factory=dict)  # entity coord 'storage'
    invest_min: xr.DataArray | None = None  # (invest_storage,)
    invest_max: xr.DataArray | None = None  # (invest_storage,)
    invest_mandatory: xr.DataArray | None = None  # (invest_storage,)
    invest_lifetime: xr.DataArray | None = None  # (invest_storage,) — NaN = forever
    invest_prior_size: xr.DataArray | None = None  # (invest_storage,)
    invest_effects_per_size_at_build: dict[str, xr.DataArray] = field(default_factory=dict)  # once, 'storage'
    invest_effects_fixed_at_build: dict[str, xr.DataArray] = field(default_factory=dict)  # once, 'storage'
    invest_effects_per_size_recurring: dict[str, xr.DataArray] = field(default_factory=dict)  # 'storage'
    invest_effects_fixed_recurring: dict[str, xr.DataArray] = field(default_factory=dict)  # 'storage'

    def __post_init__(self) -> None:
        """Validate capacity, efficiencies, loss rates, and stacked pair uniqueness."""
        for family in (
            'sizing_effects_per_size',
            'sizing_effects_fixed',
            'invest_effects_per_size_at_build',
            'invest_effects_fixed_at_build',
            'invest_effects_per_size_recurring',
            'invest_effects_fixed_recurring',
        ):
            _validate_stacked_pairs(family, getattr(self, family), 'storage')
        s = self.capacity.coords['storage']
        cap = self.capacity
        bad_cap = ~np.isnan(cap) & (cap < 0)
        if bad_cap.any():
            raise ValueError(f'Negative capacity on storages: {list(s[bad_cap].values)}')
        bad_eta_c = ((self.eta_c <= 0) | (self.eta_c > 1)).any('time')
        if bad_eta_c.any():
            raise ValueError(f'eta_charge must be in (0, 1] on storages: {list(s[bad_eta_c].values)}')
        bad_eta_d = ((self.eta_d <= 0) | (self.eta_d > 1)).any('time')
        if bad_eta_d.any():
            raise ValueError(f'eta_discharge must be in (0, 1] on storages: {list(s[bad_eta_d].values)}')
        bad_loss = ((self.loss < 0) | (self.loss > 1)).any('time')
        if bad_loss.any():
            raise ValueError(f'relative_loss_per_hour must be in [0, 1] on storages: {list(s[bad_loss].values)}')

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, children: dict[str, dict[str, xr.DataArray]] | None = None) -> Self:
        """Deserialize from xr.Dataset plus netCDF child groups.

        Args:
            ds: Dataset with matching variable names.
            children: Signature-grouped families parsed from the netCDF
                child groups, keyed by field name.
        """
        return _from_dataset_with_children(cls, ds, children)

    @classmethod
    def build(
        cls,
        storages: list[Storage],
        time: TimeIndex,
        dt: xr.DataArray,
        effects: list[Effect] | None = None,
        period: pd.Index | None = None,
    ) -> Self | None:
        """Build StoragesData from element objects.

        Args:
            storages: Storage definitions.
            time: Time index.
            dt: Timestep durations.
            effects: Effect definitions for sizing cost validation.
            period: Period index for period-varying effects.
        """
        from fluxopt.elements import Investment, Sizing

        if not storages:
            return None

        effect_ids = [e.id for e in effects] if effects else []
        stor_ids = [s.id for s in storages]
        n = len(storages)

        capacity_vals = np.full(n, np.nan)
        eta_cs: list[xr.DataArray] = []
        eta_ds: list[xr.DataArray] = []
        losses: list[xr.DataArray] = []
        level_lbs: list[xr.DataArray] = []
        level_ubs: list[xr.DataArray] = []
        prior_level_vals = np.full(n, np.nan)
        cyclic_vals = np.zeros(n, dtype=bool)
        final_min_vals = np.full(n, np.nan)
        final_max_vals = np.full(n, np.nan)
        prevent_vals = np.zeros(n, dtype=bool)
        charge_flow: list[str] = []
        discharge_flow: list[str] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []

        for i, s in enumerate(storages):
            if isinstance(s.capacity, Sizing):
                sizing_items.append((s.id, s.capacity))
            elif isinstance(s.capacity, Investment):
                invest_items.append((s.id, s.capacity))
            elif s.capacity is not None:
                capacity_vals[i] = s.capacity

            eta_cs.append(as_dataarray(s.eta_charge, {'time': time}))
            eta_ds.append(as_dataarray(s.eta_discharge, {'time': time}))
            losses.append(as_dataarray(s.relative_loss_per_hour, {'time': time}))

            level_lbs.append(as_dataarray(s.relative_level_min, {'time': time}))
            level_ubs.append(as_dataarray(s.relative_level_max, {'time': time}))

            cyclic_vals[i] = s.cyclic
            if s.prior_level is not None:
                prior_level_vals[i] = s.prior_level
            if s.final_level_min is not None:
                final_min_vals[i] = s.final_level_min
            if s.final_level_max is not None:
                final_max_vals[i] = s.final_level_max
            prevent_vals[i] = s.prevent_simultaneous

            charge_flow.append(s.charging.id)
            discharge_flow.append(s.discharging.id)

        stor_idx = pd.Index(stor_ids, name='storage')
        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_storage', entity_label='storage', period=period)
        inv = _InvestmentArrays.build(
            invest_items, effect_ids, dim='invest_storage', entity_label='storage', period=period
        )

        return cls(
            capacity=xr.DataArray(capacity_vals, dims=['storage'], coords={'storage': stor_ids}),
            eta_c=xr.concat(eta_cs, dim=stor_idx),
            eta_d=xr.concat(eta_ds, dim=stor_idx),
            loss=xr.concat(losses, dim=stor_idx),
            rel_level_lb=xr.concat(level_lbs, dim=stor_idx),
            rel_level_ub=xr.concat(level_ubs, dim=stor_idx),
            prior_level=xr.DataArray(prior_level_vals, dims=['storage'], coords={'storage': stor_ids}),
            cyclic=xr.DataArray(cyclic_vals, dims=['storage'], coords={'storage': stor_ids}),
            charge_flow=xr.DataArray(charge_flow, dims=['storage'], coords={'storage': stor_ids}),
            discharge_flow=xr.DataArray(discharge_flow, dims=['storage'], coords={'storage': stor_ids}),
            final_level_min=(
                xr.DataArray(final_min_vals, dims=['storage'], coords={'storage': stor_ids})
                if not np.all(np.isnan(final_min_vals))
                else None
            ),
            final_level_max=(
                xr.DataArray(final_max_vals, dims=['storage'], coords={'storage': stor_ids})
                if not np.all(np.isnan(final_max_vals))
                else None
            ),
            prevent_simultaneous=(
                xr.DataArray(prevent_vals, dims=['storage'], coords={'storage': stor_ids})
                if prevent_vals.any()
                else None
            ),
            sizing_min=sz.min,
            sizing_max=sz.max,
            sizing_mandatory=sz.mandatory,
            sizing_effects_per_size=sz.effects_per_size,
            sizing_effects_fixed=sz.effects_fixed,
            invest_min=inv.min,
            invest_max=inv.max,
            invest_mandatory=inv.mandatory,
            invest_lifetime=inv.lifetime,
            invest_prior_size=inv.prior_size,
            invest_effects_per_size_at_build=inv.effects_per_size_at_build,
            invest_effects_fixed_at_build=inv.effects_fixed_at_build,
            invest_effects_per_size_recurring=inv.effects_per_size_recurring,
            invest_effects_fixed_recurring=inv.effects_fixed_recurring,
        )


def _compute_period_weights(
    periods: list[int] | pd.Index,
    period_weights: list[float] | None = None,
) -> tuple[pd.Index, xr.DataArray]:
    """Compute period weights from a period index.

    Args:
        periods: Integer period labels (e.g. [2020, 2025, 2030]).
        period_weights: Explicit weights per period. If None, inferred from
            ``np.diff(periods)`` with the last gap repeated.

    Returns:
        Tuple of (period_index, period_weights DataArray).
    """
    idx = pd.Index(periods, name='period')
    if not np.issubdtype(idx.dtype, np.integer):  # pyrefly: ignore[bad-argument-type]
        raise TypeError(f'periods must be integer, got {idx.dtype}')
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError('periods must be monotonically increasing and unique')

    if period_weights is not None:
        if len(period_weights) != len(idx):
            msg = f'period_weights has {len(period_weights)} entries, expected {len(idx)}'
            raise ValueError(msg)
        w = np.asarray(period_weights, dtype=float)
    elif len(idx) < 2:
        raise ValueError('period_weights is required when only one period is given')
    else:
        gaps = np.diff(idx.to_numpy().astype(int))
        w = np.append(gaps, gaps[-1])

    if not np.all(np.isfinite(w)) or not np.all(w > 0):
        raise ValueError(f'period_weights must be positive and finite, got {w}')

    return idx, xr.DataArray(w, dims=['period'], coords={'period': idx}, name='period_weight')


@dataclass
class Dims:
    """Shared model coordinates and temporal metadata.

    Owns the time and period dimensions, timestep durations, and weights.
    """

    time: xr.DataArray  # (time,) — coordinate labels
    dt: xr.DataArray  # (time,) — timestep durations [h]
    weights: xr.DataArray  # (time,) — timestep weights
    period: xr.DataArray | None = None  # (period,) — coordinate labels
    period_weights: xr.DataArray | None = None  # (period,) — duration weights

    def __post_init__(self) -> None:
        for name, arr in [('dt', self.dt), ('weights', self.weights)]:
            if arr.dims != ('time',):
                raise ValueError(f"Dims.{name} must be 1D with dims=('time',), got {arr.dims}")
            if not arr.coords['time'].equals(self.time):
                raise ValueError(f'Dims.{name} time coordinate does not match Dims.time')

    def coords(self, *, time: bool = False, period: bool = False) -> dict[str, xr.DataArray]:
        """Return shared coordinates for variable/DataArray creation.

        Also the single point of truth for the model's variate dims used by
        :func:`fluxopt.types.as_dataarray`: pick the reach a field supports
        (e.g. ``coords(time=True, period=True)`` for operational profiles,
        ``coords(period=True)`` for investment-time fields). When a new
        variate dim (e.g. ``scenario``) is added, extend this method once
        and every call site picks it up.

        Args:
            time: Include the time coordinate.
            period: Include the period coordinate (no-op in single-period mode).
        """
        result: dict[str, xr.DataArray] = {}
        if time:
            result['time'] = self.time
        if period and self.period is not None:
            result['period'] = self.period
        return result

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        data_vars: dict[str, xr.DataArray] = {'dt': self.dt, 'weights': self.weights}
        if self.period is not None:
            data_vars['period'] = self.period
        if self.period_weights is not None:
            data_vars['period_weights'] = self.period_weights
        return xr.Dataset(data_vars)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with dt, weights, and optional period fields.
        """
        dt = ds['dt']
        time_idx = dt.coords['time']
        return cls(
            time=time_idx,
            dt=dt,
            weights=ds['weights'],
            period=ds.get('period', None),
            period_weights=ds.get('period_weights', None),
        )

    @classmethod
    def build(
        cls,
        time: TimeIndex,
        dt: xr.DataArray,
        periods: list[int] | pd.Index | None = None,
        period_weights: list[float] | None = None,
    ) -> Self:
        """Build Dims from a time index and optional periods.

        Args:
            time: Normalized time index.
            dt: Timestep durations.
            periods: Integer period labels for multi-period optimization.
            period_weights: Explicit weights per period. Inferred from gaps if None.
        """
        time_coord = xr.DataArray(time, dims=['time'], coords={'time': time})
        weights = xr.DataArray(np.ones(len(time)), dims=['time'], coords={'time': time}, name='weight')

        period_da: xr.DataArray | None = None
        period_weights_da: xr.DataArray | None = None
        if periods is not None:
            period_idx, period_weights_da = _compute_period_weights(periods, period_weights)
            period_da = xr.DataArray(period_idx.values, dims=['period'], coords={'period': period_idx})

        return cls(
            time=time_coord,
            dt=dt,
            weights=weights,
            period=period_da,
            period_weights=period_weights_da,
        )


@dataclass
class ModelData:
    flows: FlowsData
    carriers: CarriersData
    converters: ConvertersData | None  # None when no linear converters
    effects: EffectsData
    storages: StoragesData | None  # None when no storages
    dims: Dims
    piecewise: PiecewiseData | None = None  # None when no piecewise converters

    def netcdf_nodes(self) -> dict[str, xr.Dataset]:
        """Group-path -> Dataset mapping for persistence.

        One node per table under ``model/``. Dict fields (signature-grouped
        coefficients) become one ``sig_*`` child group per signature so each
        keeps its own ``contribution`` dim — every group is self-contained,
        with labeling coords under their runtime names (``flow``, ``effect``).
        """
        nodes: dict[str, xr.Dataset] = {}
        tables: dict[
            str,
            FlowsData | CarriersData | ConvertersData | EffectsData | StoragesData | PiecewiseData | None,
        ] = {
            'flows': self.flows,
            'carriers': self.carriers,
            'converters': self.converters,
            'effects': self.effects,
            'storages': self.storages,
            'piecewise': self.piecewise,
        }
        for name, obj in tables.items():
            if obj is None:
                continue
            nodes[_NC_GROUPS[name]] = obj.to_dataset()
            for fname in _dict_field_names(type(obj)):
                for signature, arr in getattr(obj, fname).items():
                    nodes[f'{_NC_GROUPS[name]}/{fname}/sig_{signature}'] = xr.Dataset({'value': arr})
        nodes['model/meta'] = self.dims.to_dataset()
        return nodes

    def to_netcdf(self, path: str | Path, *, mode: Literal['w', 'a'] = 'a') -> None:
        """Write model data as NetCDF groups under ``/model/``.

        Args:
            path: Output file path.
            mode: Mode for the first group ('w' to overwrite the file,
                'a' to append to an existing one); remaining groups append.
        """
        p = Path(path)
        for group, ds in self.netcdf_nodes().items():
            ds.to_netcdf(p, group=group, mode=mode, engine='netcdf4')
            mode = 'a'

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ModelData:
        """Read model data from NetCDF groups.

        Opens the file once and loads each group through that handle —
        per-group :func:`xr.load_dataset` calls would pay a file open per
        group.

        Args:
            path: Input file path.

        Raises:
            OSError: If no model data groups found in the file.
            ValueError: On Windows when reading a non-ASCII path (netcdf4 limitation).
        """
        import netCDF4

        p = Path(path)
        try:
            nc = netCDF4.Dataset(p, 'r')
        except OSError as e:
            _raise_netcdf_read_error(p, e)
        try:
            groups = _list_groups(nc)
            if '/model/meta' not in groups:
                raise OSError(f'No model data groups found in {p}')

            def table(name: str) -> xr.Dataset:
                group = _NC_GROUPS[name]
                return _load_group(nc, group) if f'/{group}' in groups else xr.Dataset()

            def children_of(name: str) -> dict[str, dict[str, xr.DataArray]]:
                """Collect ``sig_*`` child groups back into signature dicts."""
                group = _NC_GROUPS[name]
                out: dict[str, dict[str, xr.DataArray]] = {}
                for full in sorted(groups):
                    head, _, sig = full.rpartition('/')
                    if sig.startswith('sig_') and head.startswith(f'/{group}/'):
                        fname = head.removeprefix(f'/{group}/')
                        out.setdefault(fname, {})[sig.removeprefix('sig_')] = _load_group(nc, full)['value']
                return out

            flows = FlowsData.from_dataset(table('flows'), children=children_of('flows'))
            carriers = CarriersData.from_dataset(table('carriers'))
            converters_ds = table('converters')
            converters = ConvertersData.from_dataset(converters_ds) if converters_ds.data_vars else None
            effects = EffectsData.from_dataset(table('effects'))
            storages_ds = table('storages')
            storages = (
                StoragesData.from_dataset(storages_ds, children=children_of('storages'))
                if storages_ds.data_vars
                else None
            )
            piecewise_ds = table('piecewise')
            piecewise = PiecewiseData.from_dataset(piecewise_ds) if piecewise_ds.data_vars else None

            return cls(
                flows=flows,
                carriers=carriers,
                converters=converters,
                effects=effects,
                storages=storages,
                dims=Dims.from_dataset(_load_group(nc, 'model/meta')),
                piecewise=piecewise,
            )
        finally:
            nc.close()

    @classmethod
    def build(
        cls,
        timesteps: Timesteps,
        carriers: list[Carrier],
        effects: list[Effect],
        ports: list[Port],
        converters: list[Converter] | None = None,
        storages: list[Storage] | None = None,
        dt: float | list[float] | None = None,
        periods: list[int] | pd.Index | None = None,
        period_weights: list[float] | None = None,
    ) -> Self:
        """Build ModelData from element objects.

        Args:
            timesteps: Time index for the optimization horizon.
            carriers: Carrier declarations.
            effects: Effects to track.
            ports: System boundary ports.
            converters: Linear converters.
            storages: Energy storages.
            dt: Timestep duration in hours. Auto-derived if None.
            periods: Integer period labels for multi-period optimization.
            period_weights: Explicit weights per period. Inferred from gaps if None.
        """
        from fluxopt.elements import PENALTY_EFFECT_ID, Effect
        from fluxopt.types import compute_dt as _compute_dt

        converters = converters or []
        stor_list = storages or []
        time = normalize_timesteps(timesteps)
        dt_da = _compute_dt(time, dt)

        if not any(e.id == PENALTY_EFFECT_ID for e in effects):
            effects = [*effects, Effect(id=PENALTY_EFFECT_ID)]

        flows, carrier_coeff = _collect_flows(ports, converters, stor_list)
        _validate_system(effects, ports, converters, stor_list, flows, carriers)

        dims = Dims.build(time, dt_da, periods=periods, period_weights=period_weights)

        # Scalar dt for prior duration computation (use first timestep)
        dt_scalar = float(dims.dt.values[0])
        period_idx = pd.Index(dims.period.values) if dims.period is not None else None

        comp_status_items: list[tuple[str, Status, list[str]]] = [
            (s.id, s.status, [s.charging.id, s.discharging.id]) for s in stor_list if s.status is not None
        ]
        comp_status_items.extend(
            (c.id, c.conversion.status, [f.id for f in (*c.inputs, *c.outputs)])
            for c in converters
            if c.conversion is not None and c.conversion.status is not None
        )

        flows_data = FlowsData.build(
            flows,
            time,
            effects,
            dt=dt_scalar,
            period=period_idx,
            component_status_items=comp_status_items,
        )
        carriers_data = CarriersData.build(carriers, flows, carrier_coeff)
        converters_data = ConvertersData.build(converters, time)
        effects_data = EffectsData.build(effects, time, period=period_idx)
        storages_data = StoragesData.build(stor_list, time, dims.dt, effects, period=period_idx)
        piecewise_data = PiecewiseData.build(converters, time)

        return cls(
            flows=flows_data,
            carriers=carriers_data,
            converters=converters_data,
            effects=effects_data,
            storages=storages_data,
            dims=dims,
            piecewise=piecewise_data,
        )


def _collect_flows(
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage] | None,
) -> tuple[list[Flow], dict[str, float]]:
    """Gather all flows and assign carrier-balance coefficients by direction.

    Args:
        ports: System boundary ports.
        converters: Converter components.
        storages: Storage components.

    Returns:
        Tuple of (flows, carrier_coeff) where carrier_coeff maps flow id to
        +1 (produces into carrier) or -1 (consumes from carrier).
    """
    flows: list[Flow] = []
    carrier_coeff: dict[str, float] = {}
    for port in ports:
        for f in port.imports:
            flows.append(f)
            carrier_coeff[f.id] = 1.0  # imports add energy to carrier
        for f in port.exports:
            flows.append(f)
            carrier_coeff[f.id] = -1.0  # exports take energy from carrier
    for conv in converters:
        for f in conv.inputs:
            flows.append(f)
            carrier_coeff[f.id] = -1.0  # converter consumes from carrier
        for f in conv.outputs:
            flows.append(f)
            carrier_coeff[f.id] = 1.0  # converter produces to carrier
    for s in storages or []:
        flows.append(s.charging)
        carrier_coeff[s.charging.id] = -1.0  # charging takes from carrier
        flows.append(s.discharging)
        carrier_coeff[s.discharging.id] = 1.0  # discharging adds to carrier
    return flows, carrier_coeff


def _validate_system(
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage],
    flows: list[Flow],
    carriers: list[Carrier],
) -> None:
    """Validate unique ids and carrier consistency across all elements.

    Args:
        effects: Effect definitions.
        ports: Port components.
        converters: Converter components.
        storages: Storage components.
        flows: All collected flows.
        carriers: Declared carriers.
    """
    # Unique component IDs
    all_ids: list[str] = [e.id for e in effects]
    all_ids.extend(p.id for p in ports)
    all_ids.extend(c.id for c in converters)
    all_ids.extend(s.id for s in storages)
    seen: set[str] = set()
    for id_ in all_ids:
        if id_ in seen:
            raise ValueError(f'Duplicate id: {id_!r}')
        seen.add(id_)

    # Unique flow IDs
    flow_seen: set[str] = set()
    for flow in flows:
        if flow.id in flow_seen:
            raise ValueError(f'Duplicate flow id: {flow.id!r}')
        flow_seen.add(flow.id)

    # Unique carrier IDs
    carrier_id_list = [c.id for c in carriers]
    carrier_ids = set[str]()
    for cid in carrier_id_list:
        if cid in carrier_ids:
            raise ValueError(f'Duplicate carrier id: {cid!r}')
        carrier_ids.add(cid)

    # Every flow carrier must match a declared carrier
    carrier_by_id = {c.id: c for c in carriers}
    for flow in flows:
        if flow.carrier not in carrier_by_id:
            raise ValueError(
                f'Flow {flow.id!r} references carrier {flow.carrier!r} '
                f'which is not in the declared carriers: {sorted(carrier_by_id)}'
            )
        carrier = carrier_by_id[flow.carrier]
        if flow.node and not carrier.nodes:
            raise ValueError(f'Flow {flow.id!r} specifies node={flow.node!r} but carrier {carrier.id!r} has no nodes')
        if flow.node and flow.node not in carrier.nodes:
            raise ValueError(
                f'Flow {flow.id!r} specifies node={flow.node!r} but carrier '
                f'{carrier.id!r} only has nodes {carrier.nodes}'
            )
