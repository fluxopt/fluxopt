from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest
import xarray as xr

from fluxopt import Carrier, Converter, Effect, Flow, Port, Storage, optimize
from fluxopt.results import Result

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_nc(tmp_path: Path) -> Path:
    return tmp_path / 'result.nc'


def _solve_simple(timesteps: list[datetime] | list[int]) -> Result:
    """Simple source -> demand system with cost tracking."""
    demand = Flow('elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
    source = Flow('elec', size=200, effects_per_flow_hour={'cost': 0.04})
    return optimize(
        timesteps=timesteps,
        carriers=[Carrier('elec')],
        effects=[Effect('cost')],
        objective_effects='cost',
        ports=[Port('grid', imports=[source]), Port('demand', exports=[demand])],
    )


def _solve_with_storage(timesteps: list[datetime]) -> Result:
    """Boiler + storage system."""
    demand = Flow('heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
    gas_source = Flow('gas', size=500, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02]})
    fuel = Flow('gas', size=300)
    heat_out = Flow('heat', size=200)
    charge = Flow('heat', size=100)
    discharge = Flow('heat', size=100)
    storage = Storage('heat_store', charging=charge, discharging=discharge, capacity=200.0)
    return optimize(
        timesteps=timesteps,
        carriers=[Carrier('gas'), Carrier('heat')],
        effects=[Effect('cost')],
        objective_effects='cost',
        ports=[Port('grid', imports=[gas_source]), Port('demand', exports=[demand])],
        converters=[Converter.boiler('boiler', 0.9, fuel, heat_out)],
        storages=[storage],
    )


class TestRoundtrip:
    def test_simple_datetime(self, tmp_nc: Path) -> None:
        """Roundtrip: simple model with datetime timesteps."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.objective == pytest.approx(result.objective, abs=1e-6)

    def test_with_storage(self, tmp_nc: Path) -> None:
        """Roundtrip: model with storage."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.objective == pytest.approx(result.objective, abs=1e-6)

    def test_model_data_preserved(self, tmp_nc: Path) -> None:
        """ModelData survives a NetCDF roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)
        assert result.data is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.data is not None
        # Flows dataset preserved
        assert list(loaded.data.flows.rel_lb.coords['flow'].values) == list(
            result.data.flows.rel_lb.coords['flow'].values
        )
        # Storages dataset preserved
        assert loaded.data.storages is not None
        assert result.data.storages is not None
        assert list(loaded.data.storages.capacity.coords['storage'].values) == list(
            result.data.storages.capacity.coords['storage'].values
        )
        # Dims roundtrip: dt, time, and weights preserved with coordinates
        xr.testing.assert_equal(loaded.data.dims.dt, result.data.dims.dt)
        xr.testing.assert_equal(loaded.data.dims.time, result.data.dims.time)
        xr.testing.assert_equal(loaded.data.dims.weights, result.data.dims.weights)

    def test_model_data_resolve(self, tmp_nc: Path) -> None:
        """Loaded ModelData can build and solve a new model."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_with_storage(ts)

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)
        assert loaded.data is not None

        # Re-solve from loaded data
        from fluxopt import FlowSystem

        model = FlowSystem(loaded.data)
        result2 = model.optimize(objective_effects='cost')
        assert result2.objective == pytest.approx(result.objective, abs=1e-6)


class TestCarrierMetadataRoundtrip:
    def test_carrier_metadata_preserved(self, tmp_nc: Path) -> None:
        """Carrier unit, color, and description survive a NetCDF roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        source = Flow('elec', size=200, effects_per_flow_hour={'cost': 0.04})
        demand = Flow('elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        result = optimize(
            timesteps=ts,
            carriers=[Carrier('elec', unit='kWh', color='#ff0000', description='Electrical energy')],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[Port('grid', imports=[source]), Port('demand', exports=[demand])],
        )
        assert result.data is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.data is not None
        assert str(loaded.data.carriers.unit.sel(carrier='elec').values) == 'kWh'
        assert str(loaded.data.carriers.color.sel(carrier='elec').values) == '#ff0000'
        assert str(loaded.data.carriers.description.sel(carrier='elec').values) == 'Electrical energy'


class TestRoundtripContributionFrom:
    def test_roundtrip_with_contribution_from(self, tmp_nc: Path) -> None:
        """ModelData with contribution_from survives NetCDF roundtrip."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        source = Flow('elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow('elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts,
            carriers=[Carrier('elec')],
            effects=[
                Effect('cost', contribution_from={'co2': 50}),
                Effect('co2', unit='kg'),
            ],
            objective_effects='cost',
            ports=[Port('grid', imports=[source]), Port('demand', exports=[sink])],
        )
        assert result.data is not None
        assert result.data.effects.cf_temporal is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.data is not None
        assert loaded.data.effects.cf_temporal is not None
        xr.testing.assert_equal(loaded.data.effects.cf_temporal, result.data.effects.cf_temporal)

        # Re-solve gives same objective
        from fluxopt import FlowSystem

        model = FlowSystem(loaded.data)
        result2 = model.optimize(objective_effects='cost')
        assert result2.objective == pytest.approx(result.objective, abs=1e-6)


class TestSolutionDataset:
    def test_solution_is_dataset(self) -> None:
        """solution is an xr.Dataset with solution data."""
        ts = [datetime(2024, 1, 1, h) for h in range(3)]
        result = _solve_simple(ts)

        ds = result.solution
        assert isinstance(ds, xr.Dataset)
        assert 'flow--rate' in ds
        assert ds.attrs['objective'] == pytest.approx(result.objective)


class TestContributionsRoundtrip:
    def test_contributions_serialized(self, tmp_nc: Path) -> None:
        """Pre-computed contributions survive a NetCDF roundtrip."""
        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        assert result.contributions is not None

        result.to_netcdf(tmp_nc)
        loaded = Result.from_netcdf(tmp_nc)

        assert loaded.contributions is not None
        xr.testing.assert_allclose(loaded.contributions['temporal'], result.contributions['temporal'])
        xr.testing.assert_allclose(loaded.contributions['lump'], result.contributions['lump'])
        xr.testing.assert_allclose(loaded.contributions['total'], result.contributions['total'])

    def test_old_file_without_contributions(self, tmp_nc: Path) -> None:
        """Loading a file without contributions group falls back gracefully and warns."""
        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        # Write without contributions (simulate old format)
        result.solution.to_netcdf(tmp_nc, mode='w', engine='netcdf4')
        result.data.to_netcdf(tmp_nc)

        with pytest.warns(UserWarning, match="no 'contributions' group"):
            loaded = Result.from_netcdf(tmp_nc)
        assert loaded.contributions is None
        # Fallback re-derivation still works
        contrib = loaded.stats.effect_contributions
        assert 'temporal' in contrib

    def test_roundtrip_does_not_warn(self, tmp_nc: Path) -> None:
        """Loading a file with cached contributions does not emit the missing-group warning."""
        import warnings

        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        result.to_netcdf(tmp_nc)

        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            loaded = Result.from_netcdf(tmp_nc)
        assert loaded.contributions is not None

    def test_netcdf_group_structure(self, tmp_nc: Path) -> None:
        """The saved file has a 'contributions' NetCDF group with temporal/lump/total
        variables on the (contributor, effect[, time]) dims — verified by opening
        the group directly, independent of Result.from_netcdf."""
        result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])
        assert result.contributions is not None
        result.to_netcdf(tmp_nc)

        # Main group (solution) loads without specifying group=
        solution = xr.load_dataset(tmp_nc)
        assert 'flow--rate' in solution

        # The contributions group is its own NetCDF group on the same file.
        contrib = xr.load_dataset(tmp_nc, group='contributions')
        assert set(contrib.data_vars) == {'temporal', 'lump', 'total'}
        assert set(contrib['temporal'].dims) == {'contributor', 'effect', 'time'}
        assert set(contrib['lump'].dims) == {'contributor', 'effect'}
        assert set(contrib['total'].dims) == {'contributor', 'effect'}

    def test_from_model_warns_and_falls_back_when_compute_fails(self, monkeypatch) -> None:
        """If compute_effect_contributions raises during solve, from_model emits a
        warning and sets result.contributions to None — lazy re-derivation still
        works via the stats accessor."""
        import fluxopt.contributions as contributions_mod

        def _raise(*args, **kwargs):
            raise RuntimeError('synthetic failure for test')

        monkeypatch.setattr(contributions_mod, 'compute_effect_contributions', _raise)

        with pytest.warns(UserWarning, match='Failed to compute effect contributions'):
            result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])

        assert result.contributions is None
