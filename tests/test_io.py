from __future__ import annotations

import os
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
        from fluxopt import FlowSystemModel

        model = FlowSystemModel(loaded.data)
        result2 = model.optimize(objective_effects='cost')
        assert result2.objective == pytest.approx(result.objective, abs=1e-6)


class TestUnicodePath:
    """Reading non-ASCII netCDF paths: clarify the misleading error on Windows.

    netcdf4/libnetcdf (through 4.9.3) fails to open files under non-ASCII
    *directories* on Windows with a misleading PermissionError. On a read
    failure fluxopt replaces it with an actionable message; the guard is purely
    reactive (it only fires if netcdf4 actually raises) and read-only. Other
    platforms are unaffected. See #189 and Unidata/netcdf4-python#1482.
    """

    @pytest.mark.parametrize(
        ('os_name', 'relpath', 'clarified'),
        [
            ('nt', 'ümlaut/r.nc', True),  # Windows + non-ASCII -> clarified ValueError
            ('nt', 'ascii/r.nc', False),  # Windows + ASCII -> original error passes through
            ('posix', 'ümlaut/r.nc', False),  # other platforms work -> original error passes through
        ],
    )
    def test_read_error_clarified_only_on_windows_nonascii(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, os_name: str, relpath: str, clarified: bool
    ) -> None:
        """Read failures get a clear message only for non-ASCII paths on Windows; else propagate."""
        from fluxopt.model_data import _raise_netcdf_read_error

        monkeypatch.setattr('fluxopt.model_data.os.name', os_name)
        original = PermissionError(13, 'Permission denied')
        with pytest.raises((ValueError, OSError)) as excinfo:
            _raise_netcdf_read_error(tmp_path / relpath, original)
        if clarified:
            assert isinstance(excinfo.value, ValueError)
            assert 'non-ASCII' in str(excinfo.value)
            assert excinfo.value.__cause__ is original  # original preserved in the chain
        else:
            assert excinfo.value is original  # untouched

    @pytest.mark.skipif(os.name != 'nt', reason='upstream bug is Windows-only')
    @pytest.mark.xfail(
        strict=True,
        reason='Upstream bug: netcdf4 cannot open files in non-ASCII dirs on Windows '
        '(Unidata/netcdf4-python#1482). When this XPASSes, upstream is fixed -- drop the '
        '_raise_netcdf_read_error guard.',
    )
    def test_upstream_netcdf4_nonascii_dir_canary(self, tmp_path: Path) -> None:
        """Probe raw netcdf4 directly; alerts us (strict xfail) the day upstream fixes this."""
        from netCDF4 import Dataset  # type: ignore[import-untyped]

        d = tmp_path / 'umlaut_äöü'
        d.mkdir()
        with Dataset(str(d / 'probe.nc'), 'w') as ds:
            ds.createDimension('x', 1)
            ds.createVariable('value', 'f4', ('x',))[:] = [42]


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
        from fluxopt import FlowSystemModel

        model = FlowSystemModel(loaded.data)
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
        warning (including the exception) and sets result.contributions to None —
        lazy re-derivation via the stats accessor still produces the breakdown."""
        import fluxopt.contributions as contributions_mod

        def _raise(*args, **kwargs):
            raise RuntimeError('synthetic failure for test')

        monkeypatch.setattr(contributions_mod, 'compute_effect_contributions', _raise)

        with pytest.warns(UserWarning, match=r'synthetic failure for test'):
            result = _solve_simple([datetime(2024, 1, 1, h) for h in range(3)])

        assert result.contributions is None

        # Restore the real function and exercise the lazy fallback path:
        # result.stats.effect_contributions must still produce a valid breakdown
        # by re-deriving from solution + ModelData.
        monkeypatch.undo()
        contrib = result.stats.effect_contributions
        assert contrib is not None
        assert set(contrib.data_vars) == {'temporal', 'lump', 'total'}
