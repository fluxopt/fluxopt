"""Derived statistics from optimization results.

Each one is a reported expression in ``math/program/reporting.yaml``,
evaluated at the solve; this accessor reads it back under its old name and
puts the entities the program has no value for back in as NaN.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from fluxopt.results import Result


class StatsAccessor:
    """Post-processing statistics for a solved optimization result.

    Accessed via ``result.stats``.

    Args:
        result: Solved Result.
    """

    def __init__(self, result: Result) -> None:
        self._result = result

    def _reported(self, name: str) -> xr.DataArray:
        """A reported expression, with every declared entity present.

        An entity the expression has no row for — an unsized flow has no
        size — reads NaN, not the zero a sum over nothing would suggest.
        """
        arr = self._result.expression(name)
        data = self._result.data
        ids = {'flow': data.flows.ids, 'storage': data.storages.ids if data.storages is not None else []}
        return arr.reindex({dim: labels for dim, labels in ids.items() if dim in arr.dims})

    @cached_property
    def flow_hours(self) -> xr.DataArray:
        """Energy per flow per timestep: P_{f,t} * dt_t.

        Returns:
            DataArray (flow, time) in energy units (e.g. MWh).
        """
        return self._reported('step_flow_hours')

    @cached_property
    def total_flow_hours(self) -> xr.DataArray:
        """Total energy per flow over the horizon, weighted.

        Returns:
            DataArray (flow,) — weighted sum of flow_hours over time.
        """
        return self._reported('flow_hours')

    @cached_property
    def carrier_balance(self) -> xr.DataArray:
        """Each flow's signed contribution to its carrier — ``(flow, time)``.

        Positive produces into the carrier, negative consumes from it. The
        carrier a flow belongs to rides along as a coordinate rather than as
        an axis: a flow is on exactly one, so an axis would be mostly holes.
        Group by it to get the balance itself, which the model holds at zero::

            balance = result.stats.carrier_balance
            balance.groupby('carrier').sum()
        """
        cd = self._result.data.carriers
        balance = self._reported('carrier_balance')
        by_flow = dict(zip(cd.membership['flow'], cd.membership['carrier'], strict=True))
        flows = [str(f) for f in balance.coords['flow'].values]
        return balance.assign_coords(
            carrier=xr.DataArray([by_flow[f] for f in flows], dims=['flow'], coords={'flow': balance.coords['flow']})
        )

    @cached_property
    def effect_contributions_direct(self) -> xr.Dataset:
        """Per-contributor effect breakdown *without* cross-effect propagation.

        Each contributor carries only effects it directly emits —
        ``contribution_from`` chains are ignored. Useful for attributing
        physical quantities (raw CO2, say) without conflating them with
        priced-in monetary ones.

        Read off the program's contribution expressions as they stand: the
        coefficients are bound as declared, so these are the direct charges.

        Returns:
            Dataset with ``temporal``, ``lump``, and ``total`` DataArrays.
        """
        return self._contributions(cross_effects=False)

    @cached_property
    def effect_contributions(self) -> xr.Dataset:
        """Per-contributor effect breakdown with cross-effects propagated.

        Decomposes effect totals into per-contributor parts on a unified
        ``contributor`` dimension (flow IDs + storage IDs)::

            contrib = result.stats.effect_contributions
            contrib['temporal']  # (contributor, effect, time)
            contrib['lump']  # (contributor, effect)
            contrib['total']  # (contributor, effect) — temporal sum + lump

        Cross-effects (CO2 into cost via ``Effect.contribution_from``) are
        propagated as ``(I - C)^-1 . direct``, so each contributor is charged
        the full priced-in cost.

        Returns:
            Dataset with ``temporal``, ``lump``, and ``total`` DataArrays.

        Raises:
            ValueError: If the solve did not record a breakdown. It is read
                off the model's named expressions at solve time and cannot be
                recovered from the solution alone.
        """
        return self._contributions(cross_effects=True)

    def _contributions(self, *, cross_effects: bool) -> xr.Dataset:
        """Both breakdowns read the same stored expressions; only the propagation differs."""
        from fluxopt.contributions import contributions_from

        stored = self._result.expressions
        if not stored.data_vars:
            msg = (
                'this Result carries none of the quantities the model names, so the '
                'effect breakdown cannot be assembled: they are evaluated against a '
                'solve, and a Result without them cannot re-derive it'
            )
            raise ValueError(msg)
        return contributions_from(lambda name: stored.get(name), self._result.data, cross_effects=cross_effects)

    @cached_property
    def resolved_sizes(self) -> xr.DataArray:
        """Per-flow size resolved from declared or optimized values.

        Unlike :attr:`Result.sizes` (optimized/invested flows only), this
        carries every flow — fixed sizes as declared, invested sizes from the
        solution, and NaN for unsized flows.

        Named ``resolved_*`` rather than ``installed_*`` on purpose: advanced
        multi-period investment (#88 — cumulative builds, early retirement)
        will give "installed capacity" a precise per-period meaning
        (``cap[t] = cap[t-1] + cap_new[t] - cap_retired[t]``). This stays the
        plainer "the size value resolved for each flow" so it doesn't collide
        with that future modeling term. The computation already keys off
        ``flow--size`` (the solver's per-period in-place capacity), so it
        remains correct when #88 lands; only the value, not the API, changes.

        Returns:
            DataArray (flow,) in power units (e.g. MW).
        """
        size = self._reported('flow_size').rename('size')
        return size.where(size.coords['flow'].isin(sorted(self._result.data.flows.has_size)))

    @cached_property
    def total_duration(self) -> xr.DataArray:
        """Weighted length of the modeled horizon: Σ_t dt_t · w_t.

        The reference window for utilization metrics. Full load hours, if
        wanted, are ``capacity_factor * total_duration``.

        Returns:
            Scalar DataArray in hours.
        """
        return self._reported('total_duration')

    @cached_property
    def capacity_factor(self) -> xr.DataArray:
        """Mean utilization per flow: Σ(P·dt·w) / (size · Σ(dt·w)).

        Dimensionless in [0, 1] — energy delivered relative to running at full
        size over the whole weighted horizon. Unlike full load hours, the
        weighted duration cancels, so it is independent of horizon length and
        weight convention, and a per-period value needs no period weighting.
        NaN where the size is unknown (unsized flows) or zero.

        Returns:
            DataArray (flow[, period]) — fraction of rated capacity used.
        """
        cf = self._reported('capacity_factor')
        return cf.where(np.isfinite(cf) & self.resolved_sizes.notnull())

    @cached_property
    def resolved_capacities(self) -> xr.DataArray:
        """Per-storage energy capacity resolved from declared or optimized values.

        Storage analogue of :attr:`resolved_sizes` (see its note on the
        ``resolved_*`` naming vs #88's future "installed capacity"). Empty when
        the model has no storages.

        Returns:
            DataArray (storage,) in energy units (e.g. MWh).
        """
        if self._result.data.storages is None:
            return xr.DataArray()
        return self._reported('storage_capacity').rename('capacity')

    @cached_property
    def relative_mean_level(self) -> xr.DataArray:
        """Mean fractional charge level per storage: ⟨E⟩ / capacity.

        Time-mean level (weighted by dt·w) over the reservoir capacity — the
        storage analogue of :attr:`capacity_factor`, and the running-average
        sibling of the ``relative_level_min`` / ``relative_level_max``
        bounds. Dimensionless in [0, 1] and horizon-independent. NaN where
        capacity is unknown or zero. Empty when the model has no storages.

        Returns:
            DataArray (storage[, period]) — mean fill fraction.
        """
        if self._result.data.storages is None:
            return xr.DataArray()
        rel = self._reported('relative_mean_level')
        return rel.where(np.isfinite(rel))

    @cached_property
    def summary(self) -> xr.Dataset:
        """Headline KPIs as a named namespace.

        Composes existing accessors — :attr:`Result.objective`,
        :attr:`Result.effect_totals`, :attr:`total_duration`,
        :attr:`resolved_sizes`, :attr:`total_flow_hours`,
        :attr:`capacity_factor`, and (when the model has storages)
        :attr:`resolved_capacities` and :attr:`relative_mean_level` — into
        one Dataset. The variables live on different dimensions (``effect`` vs
        ``flow`` vs ``storage``), so access them by name: this is a KPI
        namespace, not a flat table, and ``to_dataframe()`` would broadcast
        the unrelated axes together.

        Returns:
            Dataset with ``objective``, ``effect_totals``, ``total_duration``,
            ``size``, ``total_flow_hours``, ``capacity_factor``, and — with
            storages — ``capacity`` and ``relative_mean_level``.
        """
        kpis = {
            'objective': self._result.objective,
            'effect_totals': self._result.effect_totals,
            'total_duration': self.total_duration,
            'size': self.resolved_sizes,
            'total_flow_hours': self.total_flow_hours,
            'capacity_factor': self.capacity_factor,
        }
        if self._result.data.storages is not None:
            kpis['capacity'] = self.resolved_capacities
            kpis['relative_mean_level'] = self.relative_mean_level
        return xr.Dataset(kpis)
