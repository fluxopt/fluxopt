"""Derived statistics from optimization results.

Computes post-processing quantities that require ModelData (dt, weights)
— energy totals, effect contributions, solver metadata.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

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
    def effect_contributions(self) -> xr.Dataset:
        """Per-contributor breakdown of effect contributions.

        Decomposes effect totals into per-contributor parts on a unified
        ``contributor`` dimension (flow IDs + storage IDs)::

            contrib = result.stats.effect_contributions
            contrib['temporal']  # (contributor, effect, time) — flows only
            contrib['lump']  # (contributor, effect) — flows + storages
            contrib['total']  # (contributor, effect) — temporal sum + lump

        Cross-effects (e.g. CO₂ → cost) are attributed to the originating
        contributor. The contributions are validated against solver totals;
        a ``ValueError`` is raised if they don't match.

        Returns:
            Dataset with ``temporal``, ``lump``, and ``total`` DataArrays.
        """
        from fluxopt.contributions import compute_effect_contributions

        return compute_effect_contributions(self._result.solution, self._result.data)
