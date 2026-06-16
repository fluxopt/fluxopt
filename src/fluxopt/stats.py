"""Derived statistics from optimization results.

Computes post-processing quantities that require ModelData (dt, weights)
— energy totals, effect contributions, solver metadata.
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

    @cached_property
    def flow_hours(self) -> xr.DataArray:
        """Energy per flow per timestep: P_{f,t} * dt_t.

        Returns:
            DataArray (flow, time) in energy units (e.g. MWh).
        """
        return self._result.flow_rates * self._result.data.dims.dt

    @cached_property
    def total_flow_hours(self) -> xr.DataArray:
        """Total energy per flow over the horizon, weighted.

        Returns:
            DataArray (flow,) — weighted sum of flow_hours over time.
        """
        return (self.flow_hours * self._result.data.dims.weights).sum('time')

    @cached_property
    def carrier_balance(self) -> xr.DataArray:
        """Signed balance per carrier: coeff * P. (carrier, flow, time)."""
        coeff = self._result.data.carriers.flow_coeff
        return coeff * self._result.flow_rates

    @cached_property
    def effect_contributions_direct(self) -> xr.Dataset:
        """Per-contributor effect breakdown *without* cross-effect propagation.

        Each contributor only carries effects it directly emits —
        ``contribution_from`` chains are ignored. Useful for attributing
        physical quantities (e.g. raw CO₂ emissions) without conflating them
        with priced-in monetary effects.

        Returns:
            Dataset with ``temporal``, ``lump``, and ``total`` DataArrays.
        """
        if self._result.contributions is not None:
            return self._result.contributions

        from fluxopt.contributions import compute_effect_contributions

        return compute_effect_contributions(self._result.solution, self._result.data, cross_effects=False)

    @cached_property
    def effect_contributions(self) -> xr.Dataset:
        """Per-contributor effect breakdown with cross-effects propagated.

        Decomposes effect totals into per-contributor parts on a unified
        ``contributor`` dimension (flow IDs + storage IDs)::

            contrib = result.stats.effect_contributions
            contrib['temporal']  # (contributor, effect, time)
            contrib['lump']  # (contributor, effect)
            contrib['total']  # (contributor, effect) — temporal sum + lump

        Cross-effects (e.g. CO₂ → cost via ``Effect.contribution_from``) are
        propagated through the Leontief inverse, so each contributor is
        charged the full priced-in cost. The contributions are validated
        against solver totals; a ``ValueError`` is raised if they don't match.

        Built on top of :attr:`effect_contributions_direct` — when both views
        are accessed, the heavy direct computation runs only once.

        Returns:
            Dataset with ``temporal``, ``lump``, and ``total`` DataArrays.
        """
        from fluxopt.contributions import _with_cross_effects

        return _with_cross_effects(
            self.effect_contributions_direct,
            self._result.data,
            self._result.solution,
        )

    @cached_property
    def summary(self) -> xr.Dataset:
        """Headline KPIs as a named namespace.

        Bundles the scalar objective, per-effect totals, and a per-flow view
        (size, total flow hours, full load hours) into one Dataset. The
        variables live on different dimensions (``effect`` vs ``flow``), so
        access them by name — this is a KPI namespace, not a flat table, and
        ``to_dataframe()`` would broadcast the unrelated axes together.

        Full load hours are ``Σ(P·dt·w) / size``; ``size`` and
        ``total_flow_hours`` are included so the quotient is transparent and
        unsized flows still report real throughput next to a NaN size.

        Returns:
            Dataset with ``objective``, ``effect_totals``, ``size``,
            ``total_flow_hours`` and ``full_load_hours``.
        """
        # Declared sizes (NaN for unsized/invested flows); fill the NaNs from
        # the optimized sizes. `sizes` is an empty 0-d DataArray when no flow
        # is invested, so only merge when it actually carries a `flow` dim.
        size = self._result.data.flows.size
        sizes = self._result.sizes
        if 'flow' in sizes.dims:
            size = size.fillna(sizes)

        with xr.set_options(keep_attrs=True):
            flh = (self.total_flow_hours / size).where(lambda x: np.isfinite(x))

        return xr.Dataset(
            {
                'objective': self._result.objective,
                'effect_totals': self._result.effect_totals,
                'size': size,
                'total_flow_hours': self.total_flow_hours,
                'full_load_hours': flh,
            },
        )
