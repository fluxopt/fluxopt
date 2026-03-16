from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr

from fluxopt.types import as_dataarray, fast_concat, normalize_timesteps

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Carrier, Effect, Flow, Investment, Sizing, Status, Storage
    from fluxopt.types import OnceEffectInput, TimeIndex, Timesteps


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


def _expand_once_effect(value: OnceEffectInput, period: pd.Index) -> xr.DataArray:
    """Expand an investment once-effect value to 2D (period, build_period).

    Construction rule:
        - Scalar → diagonal filled with that constant
        - list/array → treated as ``(build_period,)`` → diagonal
        - 1D DataArray ``(build_period,)`` or ``(period,)`` → diagonal
        - 2D DataArray ``(period, build_period)`` → as-is

    Args:
        value: Scalar, list, 1D, or 2D effect value.
        period: Period index (shared by both axes).
    """
    n = len(period)
    coords: dict[str, Any] = {'period': period, 'build_period': period}
    dims = ['period', 'build_period']

    if isinstance(value, (int, float)):
        return xr.DataArray(np.eye(n) * float(value), dims=dims, coords=coords)

    if isinstance(value, xr.DataArray):
        vdims = {str(d) for d in value.dims}
        if vdims == {'period', 'build_period'}:
            aligned = value.reindex(period=period, build_period=period)
            if aligned.isnull().any():
                raise ValueError(
                    f'Once-effect DataArray (period, build_period) has coords that do not '
                    f'fully cover the model periods {list(period)}'
                )
            return aligned
        if vdims <= {'period', 'build_period'} and len(vdims) == 1:
            dim_name = next(iter(vdims))
            aligned = value.reindex({dim_name: period})
            if aligned.isnull().any():
                raise ValueError(
                    f'Once-effect DataArray with dim {dim_name!r} has coords that do not '
                    f'match the model periods {list(period)}'
                )
            return xr.DataArray(np.diag(aligned.values), dims=dims, coords=coords)
        foreign = [str(d) for d in value.dims if d not in ('period', 'build_period')]
        if foreign:
            raise ValueError(
                f'Once-effect DataArray has unexpected dims {foreign}. Expected subset of (period, build_period).'
            )

    da = as_dataarray(value, {'build_period': period})
    return xr.DataArray(np.diag(da.values), dims=dims, coords=coords)


_NC_GROUPS = {
    'flows': 'model/flows',
    'carriers': 'model/carriers',
    'converters': 'model/conv',
    'effects': 'model/effects',
    'storages': 'model/stor',
}


def _to_dataset(obj: object) -> xr.Dataset:
    """Convert a data dataclass to an xr.Dataset.

    Args:
        obj: Dataclass with DataArray fields and scalar attrs.
    """
    data_vars: dict[str, xr.DataArray] = {}
    attrs: dict[str, object] = {}
    for f in fields(obj):  # type: ignore[arg-type]
        val = getattr(obj, f.name)
        if val is None:
            continue
        if isinstance(val, xr.DataArray):
            data_vars[f.name] = val
        else:
            attrs[f.name] = val
    ds = xr.Dataset(data_vars)
    ds.attrs.update(attrs)
    return ds


@dataclass
class _SizingArrays:
    min: xr.DataArray | None = None
    max: xr.DataArray | None = None
    mandatory: xr.DataArray | None = None
    effects_per_size: xr.DataArray | None = None  # (sizing_dim, effect, period?)
    effects_fixed: xr.DataArray | None = None  # (sizing_dim, effect, period?)

    def __post_init__(self) -> None:
        """Validate min >= 0 and max >= min."""
        if self.min is not None:
            mask = self.min < 0
            if mask.any():
                raise ValueError(f'Sizing.min_size < 0 on {list(self.min.coords[self.min.dims[0]][mask].values)}')
        if self.min is not None and self.max is not None:
            mask = self.max < self.min
            if mask.any():
                dim = self.min.dims[0]
                raise ValueError(f'Sizing.max_size < min_size on {list(self.min.coords[dim][mask].values)}')

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Sizing]],
        effect_ids: list[str],
        dim: str,
        period: pd.Index | None = None,
    ) -> Self:
        """Validate Sizing objects and collect into DataArrays.

        Args:
            items: Pairs of (element_id, Sizing).
            effect_ids: Known effect ids for validation.
            dim: Dimension name for the resulting arrays.
            period: Period index for period-varying effects.
        """
        if not items:
            return cls()

        effect_set = set(effect_ids)
        tmpl = _effect_template({'effect': effect_ids}, period)

        ids: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []
        mandatories: list[bool] = []
        eps_slices: list[xr.DataArray] = []
        ef_slices: list[xr.DataArray] = []

        for item_id, s in items:
            ids.append(item_id)
            mins.append(s.min_size)
            maxs.append(s.max_size)
            mandatories.append(s.mandatory)

            eps = tmpl.zeros()
            ef = tmpl.zeros()
            for ek, ev in s.effects_per_size.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Sizing.effects_per_size on {item_id!r}')
                eps.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            for ek, ev in s.effects_fixed.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Sizing.effects_fixed on {item_id!r}')
                ef.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            eps_slices.append(eps)
            ef_slices.append(ef)

        coords = {dim: ids}
        sizing_idx = pd.Index(ids, name=dim)
        return cls(
            min=xr.DataArray(np.array(mins), dims=[dim], coords=coords),
            max=xr.DataArray(np.array(maxs), dims=[dim], coords=coords),
            mandatory=xr.DataArray(np.array(mandatories), dims=[dim], coords=coords),
            effects_per_size=fast_concat(eps_slices, sizing_idx),
            effects_fixed=fast_concat(ef_slices, sizing_idx),
        )


@dataclass
class _InvestmentArrays:
    min: xr.DataArray | None = None  # (invest_dim,)
    max: xr.DataArray | None = None  # (invest_dim,)
    mandatory: xr.DataArray | None = None  # (invest_dim,)
    lifetime: xr.DataArray | None = None  # (invest_dim,) — NaN = forever
    prior_size: xr.DataArray | None = None  # (invest_dim,)
    effects_per_size: xr.DataArray | None = None  # (invest_dim, effect, period, build_period) — once
    effects_fixed: xr.DataArray | None = None  # (invest_dim, effect, period, build_period) — once
    effects_per_size_periodic: xr.DataArray | None = None  # (invest_dim, effect, period?)
    effects_fixed_periodic: xr.DataArray | None = None  # (invest_dim, effect, period?)

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Investment]],
        effect_ids: list[str],
        dim: str,
        period: pd.Index | None = None,
    ) -> Self:
        """Validate Investment objects and collect into DataArrays.

        Args:
            items: Pairs of (element_id, Investment).
            effect_ids: Known effect ids for validation.
            dim: Dimension name for the resulting arrays.
            period: Period index for period-varying effects.
        """
        if not items:
            return cls()

        effect_set = set(effect_ids)
        periodic_tmpl = _effect_template({'effect': effect_ids}, period)

        # Once-effect template: (effect, period, build_period) when multi-period
        once_coords: dict[str, Any]
        once_shape: tuple[int, ...]
        once_dims: tuple[str, ...]
        if period is not None:
            n_p = len(period)
            once_coords = {
                'effect': effect_ids,
                'period': period,
                'build_period': period,
            }
            once_shape = (len(effect_ids), n_p, n_p)
            once_dims = ('effect', 'period', 'build_period')
        else:
            once_coords = {'effect': effect_ids}
            once_shape = (len(effect_ids),)
            once_dims = ('effect',)

        ids: list[str] = []
        mins: list[float] = []
        maxs: list[float] = []
        mandatories: list[bool] = []
        lifetimes: list[float] = []
        prior_sizes: list[float] = []
        all_slices: dict[str, list[xr.DataArray]] = {
            'eps': [],
            'ef': [],
            'eps_p': [],
            'ef_p': [],
        }

        for item_id, inv in items:
            if inv.max_size < inv.min_size:
                raise ValueError(f'Investment on {item_id!r}: max_size ({inv.max_size}) < min_size ({inv.min_size})')
            if inv.prior_size < 0:
                raise ValueError(f'Investment on {item_id!r}: prior_size must be >= 0, got {inv.prior_size}')
            if inv.lifetime is not None and inv.lifetime <= 0:
                raise ValueError(f'Investment on {item_id!r}: lifetime must be positive, got {inv.lifetime}')

            ids.append(item_id)
            mins.append(inv.min_size)
            maxs.append(inv.max_size)
            mandatories.append(inv.mandatory)
            lifetimes.append(float(inv.lifetime) if inv.lifetime is not None else np.nan)
            prior_sizes.append(inv.prior_size)

            # Once-effects: expand to (effect, period, build_period) via diagonal rule
            for label, src_dict, dest_key in [
                ('Investment.effects_per_size', inv.effects_per_size, 'eps'),
                ('Investment.effects_fixed', inv.effects_fixed, 'ef'),
            ]:
                arr = xr.DataArray(np.zeros(once_shape), dims=list(once_dims), coords=once_coords)
                for ek, ev in src_dict.items():
                    if ek not in effect_set:
                        raise ValueError(f'Unknown effect {ek!r} in {label} on {item_id!r}')
                    if period is not None:
                        arr.loc[ek] = _expand_once_effect(ev, period)
                    else:
                        arr.loc[ek] = as_dataarray(ev, {})
                all_slices[dest_key].append(arr)

            # Periodic effects: (effect, period?) — no build_period axis
            for label, src_dict, dest_key in [
                ('Investment.effects_per_size_periodic', inv.effects_per_size_periodic, 'eps_p'),
                ('Investment.effects_fixed_periodic', inv.effects_fixed_periodic, 'ef_p'),
            ]:
                arr = periodic_tmpl.zeros()
                for ek, ev in src_dict.items():
                    if ek not in effect_set:
                        raise ValueError(f'Unknown effect {ek!r} in {label} on {item_id!r}')
                    arr.loc[ek] = as_dataarray(ev, periodic_tmpl.as_da_coords)
                all_slices[dest_key].append(arr)

        coords = {dim: ids}
        invest_idx = pd.Index(ids, name=dim)
        return cls(
            min=xr.DataArray(np.array(mins), dims=[dim], coords=coords),
            max=xr.DataArray(np.array(maxs), dims=[dim], coords=coords),
            mandatory=xr.DataArray(np.array(mandatories), dims=[dim], coords=coords),
            lifetime=xr.DataArray(np.array(lifetimes), dims=[dim], coords=coords),
            prior_size=xr.DataArray(np.array(prior_sizes), dims=[dim], coords=coords),
            effects_per_size=fast_concat(all_slices['eps'], invest_idx),
            effects_fixed=fast_concat(all_slices['ef'], invest_idx),
            effects_per_size_periodic=fast_concat(all_slices['eps_p'], invest_idx),
            effects_fixed_periodic=fast_concat(all_slices['ef_p'], invest_idx),
        )


@dataclass
class _StatusArrays:
    min_uptime: xr.DataArray | None = None  # (status_flow,)
    max_uptime: xr.DataArray | None = None  # (status_flow,)
    min_downtime: xr.DataArray | None = None  # (status_flow,)
    max_downtime: xr.DataArray | None = None  # (status_flow,)
    initial: xr.DataArray | None = None  # (status_flow,) — NaN = free
    effects_running: xr.DataArray | None = None  # (status_flow, effect, time, period?)
    effects_startup: xr.DataArray | None = None  # (status_flow, effect, time, period?)
    previous_uptime: xr.DataArray | None = None  # (status_flow,) — hours, NaN = no prior
    previous_downtime: xr.DataArray | None = None  # (status_flow,) — hours, NaN = no prior

    def __post_init__(self) -> None:
        """Validate durations >= 0 and max >= min where both given."""
        for name in ('min_uptime', 'max_uptime', 'min_downtime', 'max_downtime'):
            arr: xr.DataArray | None = getattr(self, name)
            if arr is not None:
                mask = (~np.isnan(arr)) & (arr < 0)
                if mask.any():
                    dim = arr.dims[0]
                    raise ValueError(f'Status.{name} < 0 on {list(arr.coords[dim][mask].values)}')

        if self.min_uptime is not None and self.max_uptime is not None:
            both = ~np.isnan(self.min_uptime) & ~np.isnan(self.max_uptime)
            bad = both & (self.max_uptime < self.min_uptime)
            if bad.any():
                dim = self.min_uptime.dims[0]
                raise ValueError(f'Status.max_uptime < min_uptime on {list(self.min_uptime.coords[dim][bad].values)}')

        if self.min_downtime is not None and self.max_downtime is not None:
            both = ~np.isnan(self.min_downtime) & ~np.isnan(self.max_downtime)
            bad = both & (self.max_downtime < self.min_downtime)
            if bad.any():
                dim = self.min_downtime.dims[0]
                raise ValueError(
                    f'Status.max_downtime < min_downtime on {list(self.min_downtime.coords[dim][bad].values)}'
                )

    @classmethod
    def build(
        cls,
        items: list[tuple[str, Status]],
        effect_ids: list[str],
        time: TimeIndex,
        dim: str,
        prior_rates_map: dict[str, list[float]] | None = None,
        dt: float = 1.0,
        period: pd.Index | None = None,
    ) -> Self:
        """Validate Status objects and collect into DataArrays.

        Args:
            items: Pairs of (flow_id, Status).
            effect_ids: Known effect ids for validation.
            time: Time index for effect arrays.
            dim: Dimension name for the resulting arrays.
            prior_rates_map: Flow id to prior flow rates (MW) before horizon.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for period-varying effects.
        """
        from fluxopt.constraints.status import compute_previous_duration

        if not items:
            return cls()

        prior_rates_map = prior_rates_map or {}
        effect_set = set(effect_ids)
        tmpl = _effect_template({'effect': effect_ids, 'time': time}, period)

        ids: list[str] = []
        min_ups: list[float] = []
        max_ups: list[float] = []
        min_downs: list[float] = []
        max_downs: list[float] = []
        initials: list[float] = []
        prev_ups: list[float] = []
        prev_downs: list[float] = []
        er_slices: list[xr.DataArray] = []
        es_slices: list[xr.DataArray] = []

        for item_id, s in items:
            ids.append(item_id)
            min_ups.append(s.min_uptime if s.min_uptime is not None else np.nan)
            max_ups.append(s.max_uptime if s.max_uptime is not None else np.nan)
            min_downs.append(s.min_downtime if s.min_downtime is not None else np.nan)
            max_downs.append(s.max_downtime if s.max_downtime is not None else np.nan)

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

            er = tmpl.zeros()
            for ek, ev in s.effects_per_running_hour.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Status.effects_per_running_hour on {item_id!r}')
                er.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            er_slices.append(er)

            es = tmpl.zeros()
            for ek, ev in s.effects_per_startup.items():
                if ek not in effect_set:
                    raise ValueError(f'Unknown effect {ek!r} in Status.effects_per_startup on {item_id!r}')
                es.loc[ek] = as_dataarray(ev, tmpl.as_da_coords)
            es_slices.append(es)

        coords = {dim: ids}
        status_idx = pd.Index(ids, name=dim)

        prev_up_arr = np.array(prev_ups)
        prev_down_arr = np.array(prev_downs)

        return cls(
            min_uptime=xr.DataArray(np.array(min_ups), dims=[dim], coords=coords),
            max_uptime=xr.DataArray(np.array(max_ups), dims=[dim], coords=coords),
            min_downtime=xr.DataArray(np.array(min_downs), dims=[dim], coords=coords),
            max_downtime=xr.DataArray(np.array(max_downs), dims=[dim], coords=coords),
            initial=xr.DataArray(np.array(initials), dims=[dim], coords=coords),
            effects_running=fast_concat(er_slices, status_idx),
            effects_startup=fast_concat(es_slices, status_idx),
            previous_uptime=xr.DataArray(prev_up_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_up_arr))
            else None,
            previous_downtime=xr.DataArray(prev_down_arr, dims=[dim], coords=coords)
            if not np.all(np.isnan(prev_down_arr))
            else None,
        )


@dataclass
class FlowsData:
    bound_type: xr.DataArray  # (flow,) — 'unsized' | 'bounded' | 'profile'
    rel_lb: xr.DataArray  # (flow, time)
    rel_ub: xr.DataArray  # (flow, time)
    fixed_profile: xr.DataArray  # (flow, time) — NaN where not fixed
    size: xr.DataArray  # (flow,) — NaN for unsized
    effect_coeff: xr.DataArray  # (flow, effect, time[, period])
    sizing_min: xr.DataArray | None = None  # (sizing_flow,)
    sizing_max: xr.DataArray | None = None  # (sizing_flow,)
    sizing_mandatory: xr.DataArray | None = None  # (sizing_flow,)
    sizing_effects_per_size: xr.DataArray | None = None  # (sizing_flow, effect, period?)
    sizing_effects_fixed: xr.DataArray | None = None  # (sizing_flow, effect, period?)
    status_min_uptime: xr.DataArray | None = None  # (status_flow,)
    status_max_uptime: xr.DataArray | None = None  # (status_flow,)
    status_min_downtime: xr.DataArray | None = None  # (status_flow,)
    status_max_downtime: xr.DataArray | None = None  # (status_flow,)
    status_initial: xr.DataArray | None = None  # (status_flow,)
    status_effects_running: xr.DataArray | None = None  # (status_flow, effect, time, period?)
    status_effects_startup: xr.DataArray | None = None  # (status_flow, effect, time, period?)
    status_previous_uptime: xr.DataArray | None = None  # (status_flow,)
    status_previous_downtime: xr.DataArray | None = None  # (status_flow,)
    invest_min: xr.DataArray | None = None  # (invest_flow,)
    invest_max: xr.DataArray | None = None  # (invest_flow,)
    invest_mandatory: xr.DataArray | None = None  # (invest_flow,)
    invest_lifetime: xr.DataArray | None = None  # (invest_flow,) — NaN = forever
    invest_prior_size: xr.DataArray | None = None  # (invest_flow,)
    invest_effects_per_size: xr.DataArray | None = None  # (invest_flow, effect, period, build_period) — once
    invest_effects_fixed: xr.DataArray | None = None  # (invest_flow, effect, period, build_period) — once
    invest_effects_per_size_periodic: xr.DataArray | None = None  # (invest_flow, effect, period?)
    invest_effects_fixed_periodic: xr.DataArray | None = None  # (invest_flow, effect, period?)

    def __post_init__(self) -> None:
        """Validate relative bounds: non-negative and lb <= ub."""
        bad_neg = (self.rel_lb < -1e-12).any('time')
        if bad_neg.any():
            raise ValueError(f'Negative lower bounds on flows: {list(self.rel_lb.coords["flow"][bad_neg].values)}')
        bad_order = (self.rel_lb > self.rel_ub + 1e-12).any('time')
        if bad_order.any():
            raise ValueError(
                f'Lower bound > upper bound on flows: {list(self.rel_lb.coords["flow"][bad_order].values)}'
            )

    def to_dataset(self) -> xr.Dataset:
        """Serialize to xr.Dataset."""
        return _to_dataset(self)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with matching variable names.
        """
        kwargs: dict[str, Any] = {f.name: ds.get(f.name) for f in fields(cls)}
        return cls(**kwargs)

    @classmethod
    def build(
        cls,
        flows: list[Flow],
        time: TimeIndex,
        effects: list[Effect],
        dt: float = 1.0,
        period: pd.Index | None = None,
    ) -> Self:
        """Build FlowsData from element objects.

        Args:
            flows: All collected flows with qualified ids.
            time: Time index.
            effects: Effect definitions for cost coefficients.
            dt: Scalar timestep duration in hours for prior duration computation.
            period: Period index for multi-period models. When provided,
                ``effect_coeff`` gains a ``period`` dimension so that
                ``effects_per_flow_hour`` values can vary across periods.
        """
        from fluxopt.elements import Investment, Sizing

        flow_ids = [f.id for f in flows]
        effect_ids = [e.id for e in effects]
        effect_set = set(effect_ids)
        n_time = len(time)
        n_effects = len(effect_ids)

        bound_type: list[str] = []
        rel_lbs: list[xr.DataArray] = []
        rel_ubs: list[xr.DataArray] = []
        profiles: list[xr.DataArray] = []
        size_vals = np.full(len(flows), np.nan)
        effect_coeffs: list[xr.DataArray] = []
        sizing_items: list[tuple[str, Sizing]] = []
        invest_items: list[tuple[str, Investment]] = []
        status_items: list[tuple[str, Status]] = []
        prior_rates_map: dict[str, list[float]] = {}

        nan_time = xr.DataArray(np.full(n_time, np.nan), dims=['time'], coords={'time': time})

        for i, f in enumerate(flows):
            rel_lbs.append(as_dataarray(f.relative_minimum, {'time': time}))
            rel_ubs.append(as_dataarray(f.relative_maximum, {'time': time}))

            if isinstance(f.size, Sizing):
                sizing_items.append((f.id, f.size))
            elif isinstance(f.size, Investment):
                invest_items.append((f.id, f.size))
            elif f.size is not None:
                size_vals[i] = float(f.size)

            if f.fixed_relative_profile is not None:
                profiles.append(as_dataarray(f.fixed_relative_profile, {'time': time}))
                bound_type.append('profile')
            elif f.size is None:
                profiles.append(nan_time)
                bound_type.append('unsized')
            else:
                profiles.append(nan_time)
                bound_type.append('bounded')

            # Effect coefficients for this flow
            ec_coords: dict[str, Any] = {'effect': effect_ids, 'time': time}
            ec_shape = [n_effects, n_time]
            ec_dims = ['effect', 'time']
            if period is not None:
                ec_coords['period'] = period
                ec_shape.append(len(period))
                ec_dims.append('period')
            ec = xr.DataArray(
                np.zeros(ec_shape),
                dims=ec_dims,
                coords=ec_coords,
            )
            as_da_coords: dict[str, Any] = {'time': time}
            if period is not None:
                as_da_coords['period'] = period
            for effect_label, factor in f.effects_per_flow_hour.items():
                if effect_label not in effect_set:
                    raise ValueError(f'Unknown effect {effect_label!r} in Flow.effects_per_flow_hour on {f.id!r}')
                ec.loc[effect_label] = as_dataarray(factor, as_da_coords)
            effect_coeffs.append(ec)

            if f.status is not None:
                status_items.append((f.id, f.status))

            if f.prior_rates is not None:
                prior_rates_map[f.id] = f.prior_rates

        flow_idx = pd.Index(flow_ids, name='flow')
        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_flow', period=period)
        inv = _InvestmentArrays.build(invest_items, effect_ids, dim='invest_flow', period=period)
        st = _StatusArrays.build(
            status_items, effect_ids, time, dim='status_flow', prior_rates_map=prior_rates_map, dt=dt, period=period
        )

        return cls(
            bound_type=xr.DataArray(bound_type, dims=['flow'], coords={'flow': flow_ids}),
            rel_lb=fast_concat(rel_lbs, flow_idx),
            rel_ub=fast_concat(rel_ubs, flow_idx),
            fixed_profile=fast_concat(profiles, flow_idx),
            size=xr.DataArray(size_vals, dims=['flow'], coords={'flow': flow_ids}),
            effect_coeff=fast_concat(effect_coeffs, flow_idx),
            sizing_min=sz.min,
            sizing_max=sz.max,
            sizing_mandatory=sz.mandatory,
            sizing_effects_per_size=sz.effects_per_size,
            sizing_effects_fixed=sz.effects_fixed,
            status_min_uptime=st.min_uptime,
            status_max_uptime=st.max_uptime,
            status_min_downtime=st.min_downtime,
            status_max_downtime=st.max_downtime,
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
            invest_effects_per_size=inv.effects_per_size,
            invest_effects_fixed=inv.effects_fixed,
            invest_effects_per_size_periodic=inv.effects_per_size_periodic,
            invest_effects_fixed_periodic=inv.effects_fixed_periodic,
        )


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

        Args:
            converters: Converter definitions.
            time: Time index.
        """
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
    min_total: xr.DataArray  # (effect,)
    max_total: xr.DataArray  # (effect,)
    min_per_hour: xr.DataArray  # (effect, time)
    max_per_hour: xr.DataArray  # (effect, time)
    is_objective: xr.DataArray  # (effect,)
    objective_effect: str
    cf_periodic: xr.DataArray | None = None  # (effect, source_effect, period?)
    cf_temporal: xr.DataArray | None = None  # (effect, source_effect, time, period?)
    period_weights_periodic: xr.DataArray | None = None  # (effect, period)
    period_weights_once: xr.DataArray | None = None  # (effect, period)

    def __post_init__(self) -> None:
        """Validate exactly one objective effect exists."""
        n_obj = int(self.is_objective.sum())
        if n_obj == 0:
            raise ValueError('No objective effect found. Include an Effect with is_objective=True.')
        if n_obj > 1:
            raise ValueError(
                f'Multiple objective effects: {list(self.is_objective.coords["effect"][self.is_objective].values)}. Only one is allowed.'
            )

    def objective_weights(
        self,
        global_period_weights: xr.DataArray | None,
    ) -> tuple[xr.DataArray | int, xr.DataArray | int]:
        """Resolve period weights for the objective effect's two domains.

        Args:
            global_period_weights: Default period weights from Dims (or None).

        Returns:
            (w_periodic, w_once) — weights for recurring and one-time domains.
            Falls back to global_period_weights / 1 when no per-effect override.
        """
        k = self.objective_effect

        if self.period_weights_periodic is not None and not self.period_weights_periodic.sel(effect=k).isnull().all():
            w_periodic: xr.DataArray | int = self.period_weights_periodic.sel(effect=k)
        elif global_period_weights is not None:
            w_periodic = global_period_weights
        else:
            w_periodic = 1

        if self.period_weights_once is not None and not self.period_weights_once.sel(effect=k).isnull().all():
            w_once: xr.DataArray | int = self.period_weights_once.sel(effect=k)
        else:
            w_once = 1

        return w_periodic, w_once

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
        return cls(**kwargs)  # type: ignore[arg-type]

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
        n_time = len(time)
        objective_effect = next(
            (e.id for e in effects if e.is_objective),
            None,
        )
        if objective_effect is None:
            raise ValueError('No objective effect found. Include an Effect with is_objective=True.')

        min_total = np.full(n, np.nan)
        max_total = np.full(n, np.nan)
        min_per_hours: list[xr.DataArray] = []
        max_per_hours: list[xr.DataArray] = []
        is_objective = np.zeros(n, dtype=bool)

        nan_time = xr.DataArray(np.full(n_time, np.nan), dims=['time'], coords={'time': time})

        has_contributions = False
        for i, e in enumerate(effects):
            if e.minimum_total is not None:
                min_total[i] = e.minimum_total
            if e.maximum_total is not None:
                max_total[i] = e.maximum_total
            min_per_hours.append(
                as_dataarray(e.minimum_per_hour, {'time': time}) if e.minimum_per_hour is not None else nan_time
            )
            max_per_hours.append(
                as_dataarray(e.maximum_per_hour, {'time': time}) if e.maximum_per_hour is not None else nan_time
            )
            is_objective[i] = e.is_objective
            if e.contribution_from or e.contribution_from_per_hour:
                has_contributions = True

        # Build cross-effect contribution arrays
        cf_periodic: xr.DataArray | None = None
        cf_temporal: xr.DataArray | None = None
        if has_contributions:
            # Self-reference check
            for e in effects:
                for src_id in (*e.contribution_from, *e.contribution_from_per_hour):
                    if src_id == e.id:
                        raise ValueError(f'Effect {e.id!r} cannot reference itself in contribution_from')

            # Cycle check
            adjacency: dict[str, list[str]] = {eid: [] for eid in effect_ids}
            for e in effects:
                for src_id in {*e.contribution_from, *e.contribution_from_per_hour}:
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in contribution_from on {e.id!r}')
                    adjacency[e.id].append(src_id)
            cycle = _detect_contribution_cycle(adjacency)
            if cycle is not None:
                raise ValueError(f'Circular contribution_from dependency: {" -> ".join(cycle)}')

            tmpl_p = _effect_template({'effect': effect_ids, 'source_effect': effect_ids}, period)
            tmpl_t = _effect_template({'effect': effect_ids, 'source_effect': effect_ids, 'time': time}, period)

            periodic_mat = tmpl_p.zeros()
            temporal_mat = tmpl_t.zeros()
            for e in effects:
                for src_id, factor in e.contribution_from.items():
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in Effect.contribution_from on {e.id!r}')
                    periodic_mat.loc[e.id, src_id] = as_dataarray(factor, tmpl_p.as_da_coords)
                    temporal_mat.loc[e.id, src_id] = as_dataarray(factor, tmpl_t.as_da_coords)
                for src_id, factor_ts in e.contribution_from_per_hour.items():
                    if src_id not in effect_set:
                        raise ValueError(f'Unknown effect {src_id!r} in Effect.contribution_from_per_hour on {e.id!r}')
                    temporal_mat.loc[e.id, src_id] = as_dataarray(factor_ts, tmpl_t.as_da_coords)
            cf_periodic = periodic_mat
            cf_temporal = temporal_mat

        effect_idx = pd.Index(effect_ids, name='effect')

        # Per-effect period weights
        pw_periodic: xr.DataArray | None = None
        pw_once: xr.DataArray | None = None
        if period is not None:
            has_pw_periodic = any(e.period_weights_periodic is not None for e in effects)
            has_pw_once = any(e.period_weights_once is not None for e in effects)
            n_periods = len(period)
            if has_pw_periodic:
                mat = np.full((n, n_periods), np.nan)
                for i, e in enumerate(effects):
                    if e.period_weights_periodic is not None:
                        if len(e.period_weights_periodic) != n_periods:
                            msg = f'Effect {e.id!r}: period_weights_periodic has {len(e.period_weights_periodic)} entries, expected {n_periods}'
                            raise ValueError(msg)
                        vals = np.asarray(e.period_weights_periodic, dtype=float)
                        if not np.all(np.isfinite(vals)) or not np.all(vals > 0):
                            msg = f'Effect {e.id!r}: period_weights_periodic must be positive and finite, got {vals}'
                            raise ValueError(msg)
                        mat[i] = vals
                pw_periodic = xr.DataArray(
                    mat, dims=['effect', 'period'], coords={'effect': effect_ids, 'period': period}
                )
            if has_pw_once:
                mat = np.full((n, n_periods), np.nan)
                for i, e in enumerate(effects):
                    if e.period_weights_once is not None:
                        if len(e.period_weights_once) != n_periods:
                            msg = f'Effect {e.id!r}: period_weights_once has {len(e.period_weights_once)} entries, expected {n_periods}'
                            raise ValueError(msg)
                        vals = np.asarray(e.period_weights_once, dtype=float)
                        if not np.all(np.isfinite(vals)) or not np.all(vals > 0):
                            msg = f'Effect {e.id!r}: period_weights_once must be positive and finite, got {vals}'
                            raise ValueError(msg)
                        mat[i] = vals
                pw_once = xr.DataArray(mat, dims=['effect', 'period'], coords={'effect': effect_ids, 'period': period})

        return cls(
            min_total=xr.DataArray(min_total, dims=['effect'], coords={'effect': effect_ids}),
            max_total=xr.DataArray(max_total, dims=['effect'], coords={'effect': effect_ids}),
            min_per_hour=fast_concat(min_per_hours, effect_idx),
            max_per_hour=fast_concat(max_per_hours, effect_idx),
            is_objective=xr.DataArray(is_objective, dims=['effect'], coords={'effect': effect_ids}),
            objective_effect=objective_effect,
            cf_periodic=cf_periodic,
            cf_temporal=cf_temporal,
            period_weights_periodic=pw_periodic,
            period_weights_once=pw_once,
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
    sizing_min: xr.DataArray | None = None  # (sizing_storage,)
    sizing_max: xr.DataArray | None = None  # (sizing_storage,)
    sizing_mandatory: xr.DataArray | None = None  # (sizing_storage,)
    sizing_effects_per_size: xr.DataArray | None = None  # (sizing_storage, effect, period?)
    sizing_effects_fixed: xr.DataArray | None = None  # (sizing_storage, effect, period?)
    invest_min: xr.DataArray | None = None  # (invest_storage,)
    invest_max: xr.DataArray | None = None  # (invest_storage,)
    invest_mandatory: xr.DataArray | None = None  # (invest_storage,)
    invest_lifetime: xr.DataArray | None = None  # (invest_storage,) — NaN = forever
    invest_prior_size: xr.DataArray | None = None  # (invest_storage,)
    invest_effects_per_size: xr.DataArray | None = None  # (invest_storage, effect, period, build_period) — once
    invest_effects_fixed: xr.DataArray | None = None  # (invest_storage, effect, period, build_period) — once
    invest_effects_per_size_periodic: xr.DataArray | None = None  # (invest_storage, effect, period?)
    invest_effects_fixed_periodic: xr.DataArray | None = None  # (invest_storage, effect, period?)

    def __post_init__(self) -> None:
        """Validate capacity, efficiencies, and loss rates."""
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
    def from_dataset(cls, ds: xr.Dataset) -> Self:
        """Deserialize from xr.Dataset.

        Args:
            ds: Dataset with matching variable names.
        """
        kwargs: dict[str, Any] = {f.name: ds.get(f.name) for f in fields(cls)}
        return cls(**kwargs)

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

            level_lbs.append(as_dataarray(s.relative_minimum_level, {'time': time}))
            level_ubs.append(as_dataarray(s.relative_maximum_level, {'time': time}))

            cyclic_vals[i] = s.cyclic
            if s.prior_level is not None:
                prior_level_vals[i] = s.prior_level

            charge_flow.append(s.charging.id)
            discharge_flow.append(s.discharging.id)

        stor_idx = pd.Index(stor_ids, name='storage')
        sz = _SizingArrays.build(sizing_items, effect_ids, dim='sizing_storage', period=period)
        inv = _InvestmentArrays.build(invest_items, effect_ids, dim='invest_storage', period=period)

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
            invest_effects_per_size=inv.effects_per_size,
            invest_effects_fixed=inv.effects_fixed,
            invest_effects_per_size_periodic=inv.effects_per_size_periodic,
            invest_effects_fixed_periodic=inv.effects_fixed_periodic,
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
    if not np.issubdtype(idx.dtype, np.integer):  # type: ignore[arg-type]
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
    converters: ConvertersData | None  # None when no converters
    effects: EffectsData
    storages: StoragesData | None  # None when no storages
    dims: Dims

    def to_netcdf(self, path: str | Path, *, mode: Literal['w', 'a'] = 'a') -> None:
        """Write model data as NetCDF groups under ``/model/``.

        Args:
            path: Output file path.
            mode: Write mode ('w' to overwrite, 'a' to append).
        """
        p = Path(path)
        dataset_fields: dict[str, FlowsData | CarriersData | ConvertersData | EffectsData | StoragesData | None] = {
            'flows': self.flows,
            'carriers': self.carriers,
            'converters': self.converters,
            'effects': self.effects,
            'storages': self.storages,
        }
        current_mode = mode
        for name, obj in dataset_fields.items():
            if obj is not None:
                obj.to_dataset().to_netcdf(p, mode=current_mode, group=_NC_GROUPS[name], engine='netcdf4')
                current_mode = 'a'
        self.dims.to_dataset().to_netcdf(p, mode=current_mode, group='model/meta', engine='netcdf4')

    @classmethod
    def from_netcdf(cls, path: str | Path) -> ModelData:
        """Read model data from NetCDF groups.

        Args:
            path: Input file path.

        Raises:
            OSError: If no model data groups found in the file.
        """
        p = Path(path)
        meta = xr.load_dataset(p, group='model/meta', engine='netcdf4')

        datasets: dict[str, xr.Dataset] = {}
        for name, group in _NC_GROUPS.items():
            try:
                datasets[name] = xr.load_dataset(p, group=group, engine='netcdf4')
            except OSError:
                datasets[name] = xr.Dataset()

        flows = FlowsData.from_dataset(datasets['flows'])
        carriers = CarriersData.from_dataset(datasets['carriers'])
        converters = ConvertersData.from_dataset(datasets['converters']) if datasets['converters'].data_vars else None
        effects = EffectsData.from_dataset(datasets['effects'])
        storages = StoragesData.from_dataset(datasets['storages']) if datasets['storages'].data_vars else None

        return cls(
            flows=flows,
            carriers=carriers,
            converters=converters,
            effects=effects,
            storages=storages,
            dims=Dims.from_dataset(meta),
        )

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
            effects = [*effects, Effect(PENALTY_EFFECT_ID)]

        flows, carrier_coeff = _collect_flows(ports, converters, stor_list)
        _validate_system(effects, ports, converters, stor_list, flows, carriers)

        dims = Dims.build(time, dt_da, periods=periods, period_weights=period_weights)

        # Scalar dt for prior duration computation (use first timestep)
        dt_scalar = float(dims.dt.values[0])
        period_idx = pd.Index(dims.period.values) if dims.period is not None else None
        flows_data = FlowsData.build(flows, time, effects, dt=dt_scalar, period=period_idx)
        carriers_data = CarriersData.build(carriers, flows, carrier_coeff)
        converters_data = ConvertersData.build(converters, time)
        effects_data = EffectsData.build(effects, time, period=period_idx)
        storages_data = StoragesData.build(stor_list, time, dims.dt, effects, period=period_idx)

        return cls(
            flows=flows_data,
            carriers=carriers_data,
            converters=converters_data,
            effects=effects_data,
            storages=storages_data,
            dims=dims,
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
