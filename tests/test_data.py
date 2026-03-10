from __future__ import annotations

import pytest
from conftest import ts

from fluxopt import Converter, Effect, Flow, ModelData, Port, Storage, optimize


class TestFlowsTable:
    def test_bounds_with_size(self):
        flow = Flow('b', size=100, relative_minimum=0.2, relative_maximum=0.8)
        data = ModelData.build(ts(3), [Effect('cost', is_objective=True)], ports=[Port('src', imports=[flow])])
        ds = data.flows
        lb = ds.rel_lb.sel(flow='src(b)').values
        ub = ds.rel_ub.sel(flow='src(b)').values
        assert list(lb) == [0.2, 0.2, 0.2]
        assert list(ub) == [0.8, 0.8, 0.8]
        assert float(ds.size.sel(flow='src(b)').values) == 100.0
        assert str(ds.bound_type.sel(flow='src(b)').values) == 'bounded'

    def test_fixed_profile(self):
        flow = Flow('b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            ts(3),
            [Effect('cost', is_objective=True)],
            ports=[Port('sink', exports=[flow])],
        )
        fixed = data.flows.fixed_profile.sel(flow='sink(b)').values
        assert list(fixed) == [0.5, 0.8, 0.6]
        assert str(data.flows.bound_type.sel(flow='sink(b)').values) == 'profile'

    def test_unsized_flow(self):
        flow = Flow('b')
        data = ModelData.build(
            ts(3),
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[flow])],
        )
        assert str(data.flows.bound_type.sel(flow='src(b)').values) == 'unsized'


class TestCarriersData:
    def test_coefficients(self):
        out_flow = Flow('b', size=100)
        in_flow = Flow('b', size=100)
        data = ModelData.build(
            ts(3),
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[out_flow]), Port('sink', exports=[in_flow])],
        )
        coeffs = data.carriers.flow_coeff
        out_coeff = float(coeffs.sel(carrier='b', flow='src(b)').values)
        in_coeff = float(coeffs.sel(carrier='b', flow='sink(b)').values)
        assert out_coeff == 1.0  # output to bus
        assert in_coeff == -1.0  # input from bus


class TestConvertersTable:
    def test_scalar_factors(self):
        fuel = Flow('gas', size=200)
        heat_flow = Flow('heat', size=100)
        boiler = Converter.boiler('boiler', 0.9, fuel, heat_flow)
        data = ModelData.build(
            ts(3),
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[Flow('gas', size=200)])],
            converters=[boiler],
        )
        ds = data.converters
        assert ds is not None
        fuel_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(gas)', time=data.time[0]).values
        )
        heat_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(heat)', time=data.time[0]).values
        )
        assert fuel_coeff == 0.9
        assert heat_coeff == -1.0


class TestEffectsTable:
    def test_flow_coefficients(self):
        flow = Flow('b', size=100, effects_per_flow_hour={'cost': 0.04})
        data = ModelData.build(
            ts(3),
            [Effect('cost', is_objective=True)],
            ports=[Port('src', imports=[flow])],
        )
        coeff = data.flows.effect_coeff.sel(flow='src(b)', effect='cost')
        assert all(v == 0.04 for v in coeff.values)

    def test_objective_effect(self):
        data = ModelData.build(
            ts(3),
            [Effect('cost', is_objective=True), Effect('co2')],
            ports=[Port('src', imports=[Flow('b', size=100)])],
        )
        assert data.effects.objective_effect == 'cost'


class TestFlowNodeId:
    def test_node_included_in_default_id(self):
        """Flow with node set auto-generates carrier:node id."""
        f = Flow('heat', node='A')
        assert f.id == 'heat:A'

    def test_node_without_node_uses_carrier(self):
        """Flow without node uses carrier as id."""
        f = Flow('heat')
        assert f.id == 'heat'


class TestStorageValidation:
    def test_mismatched_carriers_raises(self):
        """Storage with different charging/discharging carriers raises ValueError."""
        with pytest.raises(ValueError, match='charging carrier'):
            Storage('bat', Flow('elec'), Flow('heat'))


class TestCarrierBalance:
    def test_carrier_balance_property(self):
        """StatsAccessor.carrier_balance returns signed balance per carrier."""
        result = optimize(
            timesteps=ts(3),
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port('src', imports=[Flow('elec', size=100, effects_per_flow_hour={'cost': 0.04})]),
                Port('sink', exports=[Flow('elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        balance = result.stats.carrier_balance
        assert 'carrier' in balance.dims
        assert 'flow' in balance.dims
        # Source has positive coeff, sink negative — balance should sum to ~0
        total = balance.sum('flow')
        for val in total.sel(carrier='elec').values:
            assert val == pytest.approx(0.0, abs=1e-6)
