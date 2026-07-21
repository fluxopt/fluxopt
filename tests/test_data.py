from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from conftest import ts

from fluxopt import Carrier, Converter, Dims, Effect, Flow, ModelData, Port, Storage, optimize


class TestFlowsTable:
    def test_bounds_with_size(self):
        flow = Flow(carrier='b', size=100, relative_rate_min=0.2, relative_rate_max=0.8)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        ds = data.flows
        lb = ds.rel_lb.sel(flow='src(b)').values
        ub = ds.rel_ub.sel(flow='src(b)').values
        assert list(lb) == [0.2, 0.2, 0.2]
        assert list(ub) == [0.8, 0.8, 0.8]
        assert float(ds.size.sel(flow='src(b)').values) == 100.0
        assert str(ds.bound_type.sel(flow='src(b)').values) == 'bounded'

    def test_fixed_profile(self):
        flow = Flow(carrier='b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='sink', exports=[flow])],
        )
        fixed = data.flows.fixed_profile.sel(flow='sink(b)').values
        assert list(fixed) == [0.5, 0.8, 0.6]
        assert str(data.flows.bound_type.sel(flow='sink(b)').values) == 'profile'

    def test_unsized_flow(self):
        flow = Flow(carrier='b')
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        assert str(data.flows.bound_type.sel(flow='src(b)').values) == 'unsized'


class TestCarriersData:
    def test_coefficients(self):
        out_flow = Flow(carrier='b', size=100)
        in_flow = Flow(carrier='b', size=100)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[out_flow]), Port(id='sink', exports=[in_flow])],
        )
        coeffs = data.carriers.flow_coeff
        out_coeff = float(coeffs.sel(carrier='b', flow='src(b)').values)
        in_coeff = float(coeffs.sel(carrier='b', flow='sink(b)').values)
        assert out_coeff == 1.0  # output to carrier
        assert in_coeff == -1.0  # input from carrier

    def test_metadata(self):
        data = ModelData.build(
            ts(2),
            carriers=[Carrier(id='elec', unit='kWh', color='blue', description='Electricity')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='elec', size=100)])],
        )
        assert str(data.carriers.unit.sel(carrier='elec').values) == 'kWh'
        assert str(data.carriers.color.sel(carrier='elec').values) == 'blue'
        assert str(data.carriers.description.sel(carrier='elec').values) == 'Electricity'

    def test_from_dataset_roundtrip(self):
        from fluxopt.model_data import CarriersData

        data = ModelData.build(
            ts(2),
            carriers=[Carrier(id='elec', unit='kWh', color='red', description='Power')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='elec', size=100)])],
        )
        ds = data.carriers.to_dataset()
        loaded = CarriersData.from_dataset(ds)
        assert str(loaded.unit.sel(carrier='elec').values) == 'kWh'
        assert str(loaded.color.sel(carrier='elec').values) == 'red'
        assert str(loaded.description.sel(carrier='elec').values) == 'Power'


class TestConvertersTable:
    def test_scalar_factors(self):
        fuel = Flow(carrier='gas', size=200)
        heat_flow = Flow(carrier='heat', size=100)
        boiler = Converter.boiler('boiler', 0.9, fuel, heat_flow)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='gas', size=200)])],
            converters=[boiler],
        )
        ds = data.converters
        assert ds is not None
        fuel_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(gas)', time=data.dims.time[0]).values
        )
        heat_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(heat)', time=data.dims.time[0]).values
        )
        assert fuel_coeff == 0.9
        assert heat_coeff == -1.0


class TestEffectsTable:
    def test_flow_coefficients(self):
        flow = Flow(carrier='b', size=100, effects_per_flow_hour={'cost': 0.04})
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        coeff = data.flows.effect_coeff.sel(flow='src(b)', effect='cost')
        assert all(v == 0.04 for v in coeff.values)


class TestFlowNodeId:
    def test_node_included_in_default_id(self):
        """Flow with node set auto-generates carrier:node id."""
        f = Flow(carrier='heat', node='A')
        assert f.id == 'heat:A'

    def test_node_without_node_uses_carrier(self):
        """Flow without node uses carrier as id."""
        f = Flow(carrier='heat')
        assert f.id == 'heat'


class TestStorageValidation:
    def test_mismatched_carriers_raises(self):
        """Storage with different charging/discharging carriers raises ValueError."""
        with pytest.raises(ValueError, match='charging carrier'):
            Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='heat'))

    def test_same_short_id_renamed_to_charge_discharge(self):
        """Storage with same short_id renames both short_id and id."""
        s = Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='elec'))
        assert s.charging.short_id == 'charge'
        assert s.discharging.short_id == 'discharge'
        assert s.charging.id == 'bat(charge)'
        assert s.discharging.id == 'bat(discharge)'

    def test_distinct_short_ids_preserved(self):
        """Storage with explicit different short_ids keeps them in qualified id."""
        s = Storage(
            id='bat', charging=Flow(carrier='elec', short_id='in'), discharging=Flow(carrier='elec', short_id='out')
        )
        assert s.charging.id == 'bat(in)'
        assert s.discharging.id == 'bat(out)'


class TestConverterValidation:
    def test_unknown_short_id_in_conversion_factors_raises(self):
        with pytest.raises(ValueError, match=r"unknown flow short_ids \['gas'\]"):
            Converter(
                id='boiler',
                inputs=[Flow(carrier='Gas')],
                outputs=[Flow(carrier='Heat')],
                conversion_factors=[{'gas': 0.9, 'Heat': -1}],
            )

    def test_unknown_short_id_reports_equation_index(self):
        with pytest.raises(ValueError, match=r'conversion_factors\[1\]'):
            Converter(
                id='chp',
                inputs=[Flow(carrier='Gas')],
                outputs=[Flow(carrier='Heat'), Flow(carrier='Elec')],
                conversion_factors=[
                    {'Gas': 0.5, 'Heat': -1},
                    {'Gas': 0.4, 'Electricity': -1},
                ],
            )

    def test_known_short_ids_pass(self):
        conv = Converter(
            id='boiler',
            inputs=[Flow(carrier='Gas')],
            outputs=[Flow(carrier='Heat')],
            conversion_factors=[{'Gas': 0.9, 'Heat': -1}],
        )
        assert conv.conversion_factors[0]['Gas'] == 0.9


class TestCarrierValidation:
    def test_undeclared_carrier_raises(self):
        """Flow referencing an undeclared carrier raises ValueError."""
        with pytest.raises(ValueError, match='not in the declared carriers'):
            optimize(
                timesteps=ts(2),
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                objective='cost',
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_undeclared_carrier_in_model_data_build(self):
        """ModelData.build rejects flows with undeclared carriers."""
        with pytest.raises(ValueError, match="carrier 'elec'"):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_duplicate_carrier_raises(self):
        """Duplicate carrier declarations raise ValueError."""
        with pytest.raises(ValueError, match='Duplicate carrier id'):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='elec'), Carrier(id='elec')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_flow_node_on_nodeless_carrier_raises(self):
        """Flow with node on a carrier without nodes raises ValueError."""
        with pytest.raises(ValueError, match='has no nodes'):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='heat')],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='A', size=100)])],
            )

    def test_flow_node_not_in_carrier_nodes_raises(self):
        """Flow with node not declared on carrier raises ValueError."""
        with pytest.raises(ValueError, match="node='C'"):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='heat', nodes=['A', 'B'])],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='C', size=100)])],
            )


class TestCarrierBalance:
    def test_carrier_balance_property(self):
        """StatsAccessor.carrier_balance returns signed balance per carrier."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='src', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04})]),
                Port(id='sink', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        balance = result.stats.carrier_balance
        assert 'carrier' in balance.dims
        assert 'flow' in balance.dims
        # Source has positive coeff, sink negative — balance should sum to ~0
        total = balance.sum('flow')
        for val in total.sel(carrier='elec').values:
            assert val == pytest.approx(0.0, abs=1e-6)


class TestMultiNodeCarrier:
    def test_independent_node_balance(self):
        """Two flows on the same carrier but different nodes get independent balance equations."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='heat', nodes=['A', 'B'])],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='src_a', imports=[Flow(carrier='heat', node='A', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='src_b', imports=[Flow(carrier='heat', node='B', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='sink_a',
                    exports=[Flow(carrier='heat', node='A', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])],
                ),
                Port(
                    id='sink_b',
                    exports=[Flow(carrier='heat', node='B', size=100, fixed_relative_profile=[0.8, 0.8, 0.8])],
                ),
            ],
        )
        # Source A matches sink A demand (50 MW)
        rate_a = result.flow_rate('src_a(heat:A)').values
        for val in rate_a:
            assert val == pytest.approx(50.0, abs=1e-4)

        # Source B matches sink B demand (80 MW)
        rate_b = result.flow_rate('src_b(heat:B)').values
        for val in rate_b:
            assert val == pytest.approx(80.0, abs=1e-4)

    def test_node_in_carrier_dim_id(self):
        """Carrier dimension coordinates contain 'heat:A' and 'heat:B'."""
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='heat', nodes=['A', 'B'])],
            effects=[Effect(id='cost')],
            ports=[
                Port(
                    id='src_a', imports=[Flow(carrier='heat', node='A', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='src_b', imports=[Flow(carrier='heat', node='B', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='sink_a',
                    exports=[Flow(carrier='heat', node='A', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])],
                ),
                Port(
                    id='sink_b',
                    exports=[Flow(carrier='heat', node='B', size=100, fixed_relative_profile=[0.8, 0.8, 0.8])],
                ),
            ],
        )
        carrier_ids = list(data.carriers.flow_coeff.coords['carrier'].values)
        assert 'heat:A' in carrier_ids
        assert 'heat:B' in carrier_ids
        assert len(carrier_ids) == 2


class TestDimsValidation:
    def test_mismatched_dim_raises(self):
        """Dims rejects arrays that are not 1D with dims=('time',)."""
        time = xr.DataArray([0, 1], dims=['time'], coords={'time': [0, 1]})
        bad_dt = xr.DataArray([1.0, 1.0], dims=['other'])
        with pytest.raises(ValueError, match='must be 1D'):
            Dims(time=time, dt=bad_dt, weights=time)

    def test_mismatched_coords_raises(self):
        """Dims rejects arrays with different time coordinates."""
        time = xr.DataArray([0, 1], dims=['time'], coords={'time': [0, 1]})
        dt = xr.DataArray([1.0, 1.0], dims=['time'], coords={'time': [0, 1]})
        bad_weights = xr.DataArray(np.ones(3), dims=['time'], coords={'time': [0, 1, 2]})
        with pytest.raises(ValueError, match='does not match'):
            Dims(time=time, dt=dt, weights=bad_weights)
