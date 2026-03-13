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

        Returns:
            Dataset with ``temporal`` (contributor, effect, time),
            ``periodic`` (contributor, effect), and ``total``
            (contributor, effect).
        """
        from fluxopt.contributions import compute_effect_contributions

        return compute_effect_contributions(self._result.solution, self._result.data)
