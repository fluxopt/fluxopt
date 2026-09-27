"""`ModelData` persists as a directory of tables, and a reloaded one solves as the original.

A solved answer is specsolve's to persist: `optimize(archive=...)` and
`specsolve.load_archive`, tested in `test_sources.py`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
import xarray as xr
from conftest import solve_data

from fluxopt import Carrier, Converter, Effect, Flow, FlowSystem, ModelData, Port, Storage

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def tmp_nc(tmp_path: Path) -> Path:
    """Where a `ModelData` is written — a directory of tables."""
    return tmp_path / 'data'


def _hours(n: int) -> list[datetime]:
    return [datetime(2024, 1, 1, h) for h in range(n)]


def _with_storage() -> FlowSystem:
    """Boiler + storage system."""
    demand = Flow(carrier='heat', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
    gas_source = Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': [0.02, 0.08, 0.02]})
    fuel = Flow(carrier='gas', size=300)
    heat_out = Flow(carrier='heat', size=200)
    charge = Flow(carrier='heat', size=100)
    discharge = Flow(carrier='heat', size=100)
    storage = Storage(id='heat_store', charging=charge, discharging=discharge, capacity=200.0)
    return FlowSystem(
        timesteps=_hours(3),
        carriers=[Carrier(id='gas'), Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[gas_source]), Port(id='demand', exports=[demand])],
        converters=[Converter.boiler('boiler', 0.9, fuel, heat_out)],
        storages=[storage],
    )


class TestRoundtrip:
    def test_model_data_preserved(self, tmp_nc: Path) -> None:
        data = _with_storage().build_data()
        data.save(tmp_nc)
        loaded = ModelData.load(tmp_nc)

        assert loaded.flows.ids == data.flows.ids
        assert loaded.storages is not None
        assert data.storages is not None
        assert loaded.storages.ids == data.storages.ids
        xr.testing.assert_equal(loaded.dims.dt, data.dims.dt)
        xr.testing.assert_equal(loaded.dims.time, data.dims.time)
        xr.testing.assert_equal(loaded.dims.weights, data.dims.weights)

    def test_reloaded_data_solves_as_the_original(self, tmp_nc: Path) -> None:
        system = _with_storage()
        data = system.build_data()
        data.save(tmp_nc)
        assert solve_data(ModelData.load(tmp_nc)).objective == pytest.approx(system.optimize().objective, abs=1e-6)


class TestCarrierMetadataRoundtrip:
    def test_carrier_metadata_preserved(self, tmp_nc: Path) -> None:
        """Carrier unit, color, and description survive a roundtrip."""
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            _hours(3),
            carriers=[Carrier(id='elec', unit='kWh', color='#ff0000', description='Electrical energy')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
        )
        data.save(tmp_nc)
        elec = ModelData.load(tmp_nc).carriers.carriers.filter(pl.col('carrier') == 'elec')
        assert (elec['unit'][0], elec['color'][0], elec['description'][0]) == ('kWh', '#ff0000', 'Electrical energy')


class TestRoundtripContributionFrom:
    def test_roundtrip_with_contribution_from(self, tmp_nc: Path) -> None:
        source = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        system = FlowSystem(
            timesteps=_hours(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost', contribution_from={'co2': 50}), Effect(id='co2', unit='kg')],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )
        data = system.build_data()
        assert not data.effects.contributions.is_empty()
        data.save(tmp_nc)
        loaded = ModelData.load(tmp_nc)

        assert loaded.effects.contributions.equals(data.effects.contributions)
        assert solve_data(loaded).objective == pytest.approx(system.optimize().objective, abs=1e-6)


class TestBuildValidation:
    def test_undeclared_effect_rejected_without_flow_system(self) -> None:
        """The raw ModelData.build path rejects undeclared effect references."""
        from fluxopt import ModelData

        with pytest.raises(ValueError, match=r"undeclared effect\(s\) \['co2'\]"):
            ModelData.build(
                [datetime(2024, 1, 1, h) for h in range(3)],
                carriers=[Carrier(id='elec')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=10, effects_per_flow_hour={'co2': 1.0})])],
            )


class TestWaistGuards:
    """Reload guards: a tampered file never passed the element or system layer.

    Each edits the container's own file inside the saved directory — which is
    what "hand-edited" looks like now that a Result is a directory of tables.
    """

    def _status_system_path(self, tmp_nc: Path) -> Path:
        from fluxopt import ModelData, Status

        boiler_fuel = Flow(carrier='elec', size=100, relative_rate_min=0.3, status=Status())
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            [datetime(2024, 1, 1, h) for h in range(3)],
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[boiler_fuel]), Port(id='demand', exports=[demand])],
        )
        data.save(tmp_nc)
        return tmp_nc

    def test_zeroed_status_lower_bound_rejected_on_load(self, tmp_nc: Path) -> None:
        """rel_lb = 0 on a status flow would make on/off degenerate; load fails loudly."""
        import polars as pl

        from fluxopt import ModelData

        p = self._status_system_path(tmp_nc)
        table = p / 'flows' / 'envelope.parquet'
        pl.read_parquet(table).with_columns(pl.lit(0.0).alias('relative_rate_min')).write_parquet(table)

        with pytest.raises(ValueError, match='on/off is indistinguishable'):
            ModelData.load(p)

    def test_nan_size_on_ramp_flow_rejected_on_load(self, tmp_nc: Path) -> None:
        """Removing the size under a ramp-limited flow fails at load, not as NaN math."""
        import polars as pl

        from fluxopt import ModelData

        source = Flow(carrier='elec', size=100, ramp_up_per_hour=0.5)
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            [datetime(2024, 1, 1, h) for h in range(3)],
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
        )
        data.save(tmp_nc)
        table = tmp_nc / 'flows' / 'sizes.parquet'
        pl.read_parquet(table).clear().write_parquet(table)

        with pytest.raises(ValueError, match='ramp_up requires a sized flow'):
            ModelData.load(tmp_nc)

    def test_dangling_storage_flow_reference_rejected_on_load(self, tmp_nc: Path) -> None:
        """A charge_flow naming a nonexistent flow fails at load, not as a KeyError at build."""
        import polars as pl

        from fluxopt import ModelData, Storage

        charge = Flow(carrier='elec', size=50)
        discharge = Flow(carrier='elec', size=50)
        source = Flow(carrier='elec', size=100)
        demand = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            [datetime(2024, 1, 1, h) for h in range(3)],
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
            storages=[Storage(id='bat', charging=charge, discharging=discharge, capacity=100.0)],
        )
        data.save(tmp_nc)
        table = tmp_nc / 'storages' / 'storages.parquet'
        pl.read_parquet(table).with_columns(pl.lit('bat(gone)').alias('charge_flow')).write_parquet(table)

        with pytest.raises(
            ValueError, match=r"storages\.charge_flow references unknown flow id\(s\) \['bat\(gone\)'\]"
        ):
            ModelData.load(tmp_nc)
