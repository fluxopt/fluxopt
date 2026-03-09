from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr

try:
    from fluxopt_plot.accessor import PlotAccessor  # type: ignore[import-not-found]
except ImportError:
    PlotAccessor = None

if TYPE_CHECKING:
    from fluxopt.model import FlowSystem
    from fluxopt.model_data import ModelData
    from fluxopt.stats import StatsAccessor


@dataclass
class Result:
    solution: xr.Dataset
    data: ModelData | None = field(default=None, repr=False)

    @property
    def objective(self) -> float:
        """Objective function value."""
        return float(self.solution.attrs['objective'])

    @property
    def flow_rates(self) -> xr.DataArray:
        """All flow rates as (flow, time) DataArray."""
        return self.solution['flow--rate']

    @property
    def storage_levels(self) -> xr.DataArray:
        """All storage levels as (storage, time) DataArray."""
        return self.solution['storage--level'] if 'storage--level' in self.solution else xr.DataArray()

    @property
    def sizes(self) -> xr.DataArray:
        """Optimized flow sizes as (flow,) DataArray."""
        return self.solution['flow--size'] if 'flow--size' in self.solution else xr.DataArray()

    @property
    def storage_capacities(self) -> xr.DataArray:
        """Optimized storage capacities as (storage,) DataArray."""
        return self.solution['storage--capacity'] if 'storage--capacity' in self.solution else xr.DataArray()

    @property
    def effect_totals(self) -> xr.DataArray:
        """Total effect values as (effect,) DataArray."""
        return self.solution['effect--total']

    @property
    def effects_temporal(self) -> xr.DataArray:
        """Per-timestep effect values as (effect, time) DataArray."""
        return self.solution['effect--temporal']

    @property
    def effects_periodic(self) -> xr.DataArray:
        """Per-period (investment) effect values as (effect,) DataArray."""
        return self.solution['effect--periodic']

    @property
    def bus_surplus(self) -> xr.DataArray:
        """Bus surplus slack (bus, time) — overproduction absorbed by penalty."""
        return self.solution['bus--surplus'] if 'bus--surplus' in self.solution else xr.DataArray()

    @property
    def bus_shortage(self) -> xr.DataArray:
        """Bus shortage slack (bus, time) — deficit covered by penalty."""
        return self.solution['bus--shortage'] if 'bus--shortage' in self.solution else xr.DataArray()

    def flow_rate(self, flow_id: str) -> xr.DataArray:
        """Get flow rate time series for a single flow.

        Args:
            flow_id: Qualified flow id.
        """
        return self.flow_rates.sel(flow=flow_id)

    def storage_level(self, storage_id: str) -> xr.DataArray:
        """Get charge state time series for a single storage.

        Args:
            storage_id: Storage id.
        """
        return self.storage_levels.sel(storage=storage_id)

    @cached_property
    def stats(self) -> StatsAccessor:
        """Post-processing statistics accessor.

        Raises:
            ValueError: If ``data`` is not available on this Result.
        """
        from fluxopt.stats import StatsAccessor

        return StatsAccessor(self)

    def to_netcdf(self, path: str | Path) -> None:
        """Write solution and model data to NetCDF.

        Args:
            path: Output file path.
        """
        p = Path(path)
        self.solution.to_netcdf(p, mode='w', engine='netcdf4')
        if self.data is not None:
            self.data.to_netcdf(p)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> Result:
        """Read a Result from a NetCDF file.

        Args:
            path: Input file path.
        """
        from fluxopt.model_data import ModelData

        p = Path(path)
        solution = xr.load_dataset(p, engine='netcdf4')
        data = ModelData.from_netcdf(p)
        return cls(solution=solution, data=data)

    @cached_property
    def plot(self) -> PlotAccessor:
        """Plotting accessor (requires ``fluxopt-plot``)."""
        if PlotAccessor is None:
            raise ImportError('Plotting requires fluxopt-plot. Install it with: pip install fluxopt-plot')
        return PlotAccessor(self)

    @classmethod
    def from_model(cls, model: FlowSystem) -> Result:
        """Extract solution from a solved linopy model.

        Args:
            model: Solved FlowSystem instance.
        """
        sol_vars: dict[str, xr.DataArray] = {
            'flow--rate': model.flow_rate.solution,
            'effect--total': model.effect_total.solution,
            'effect--temporal': model.effect_temporal.solution,
            'effect--periodic': model.effect_periodic.solution,
        }

        if model.storage_level is not None:
            sol_vars['storage--level'] = model.storage_level.solution
        if model.flow_size is not None:
            sol_vars['flow--size'] = model.flow_size.solution
        if model.flow_size_indicator is not None:
            sol_vars['flow--size_indicator'] = model.flow_size_indicator.solution
        if model.storage_capacity is not None:
            sol_vars['storage--capacity'] = model.storage_capacity.solution
        if model.storage_capacity_indicator is not None:
            sol_vars['storage--size_indicator'] = model.storage_capacity_indicator.solution
        if model.flow_on is not None:
            sol_vars['flow--on'] = model.flow_on.solution
        if model.flow_startup is not None:
            sol_vars['flow--startup'] = model.flow_startup.solution
        if model.flow_shutdown is not None:
            sol_vars['flow--shutdown'] = model.flow_shutdown.solution
        if hasattr(model, 'bus_surplus'):
            sol_vars['bus--surplus'] = model.bus_surplus.solution
        if hasattr(model, 'bus_shortage'):
            sol_vars['bus--shortage'] = model.bus_shortage.solution

        # Include custom variables added after build()
        for var_name in model.m.variables:
            if var_name not in model._builtin_var_names and var_name not in sol_vars:
                sol_vars[var_name] = model.m.variables[var_name].solution

        raw = model.m.objective.value
        obj_val = float(raw) if raw is not None else 0.0

        solution = xr.Dataset(sol_vars, attrs={'objective': obj_val})
        return cls(solution=solution, data=model.data)
