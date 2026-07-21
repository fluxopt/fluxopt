"""FlowSystem: declarative construction, YAML IO, ProfileRef resolution."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from fluxopt import (
    Carrier,
    Effect,
    Flow,
    FlowSystem,
    Port,
    ProfileRef,
)


def _merit_order_spec(demand: object) -> FlowSystem:
    """Two priced sources meeting a fixed heat demand (see test_bus.py)."""
    return FlowSystem(
        timesteps=[0, 1],
        carriers=[Carrier(id='Heat')],
        effects=[Effect(id='cost')],
        objective_effects='cost',
        ports=[
            Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=demand)]),
            Port(id='Src1', imports=[Flow(carrier='Heat', size=20, effects_per_flow_hour={'cost': 1})]),
            Port(id='Src2', imports=[Flow(carrier='Heat', size=20, effects_per_flow_hour={'cost': 2})]),
        ],
    )


class TestPythonConstruction:
    def test_optimize_matches_free_function(self) -> None:
        result = _merit_order_spec(np.array([30, 30])).optimize()
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)
        assert_allclose(result.flow_rate('Src1(Heat)').values, [20, 20], rtol=1e-5)
        assert_allclose(result.flow_rate('Src2(Heat)').values, [10, 10], rtol=1e-5)


class TestRoundTrip:
    def test_dict_roundtrip_solves_identically(self) -> None:
        spec = _merit_order_spec([30, 30])
        rebuilt = FlowSystem.from_dict(spec.to_dict())
        assert_allclose(rebuilt.optimize().effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)

    def test_yaml_roundtrip(self, tmp_path) -> None:
        spec = _merit_order_spec([30, 30])
        path = tmp_path / 'system.yaml'
        spec.to_yaml(path)
        rebuilt = FlowSystem.from_yaml(path)
        assert_allclose(rebuilt.optimize().effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)


class TestProfileRefResolution:
    def _sources(self, values: list[float]) -> dict[str, dict[str, xr.DataArray]]:
        return {'load': {'demand': xr.DataArray(values, dims=['time'])}}

    def test_ref_resolved_from_sources(self) -> None:
        spec = _merit_order_spec(ProfileRef(source='load', variable='demand'))
        result = spec.optimize(sources=self._sources([30, 30]))
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)

    def test_spec_reusable_across_sources(self) -> None:
        # Resolution runs on a copy, so the same spec solves with different data.
        spec = _merit_order_spec(ProfileRef(source='load', variable='demand'))
        c_low = spec.optimize(sources=self._sources([10, 10])).effect_totals.sel(effect='cost').item()
        c_high = spec.optimize(sources=self._sources([30, 30])).effect_totals.sel(effect='cost').item()
        assert c_low == pytest.approx(20.0)  # Src1 @1 covers 10 for 2h
        assert c_high == pytest.approx(80.0)  # Src1 @1 x20 + Src2 @2 x10, for 2h
        # The spec itself still carries the ProfileRef (not consumed).
        ref = spec.to_dict()['ports'][0]['exports'][0]['fixed_relative_profile']
        assert ref == {'source': 'load', 'variable': 'demand', 'dim': 'time'}

    def test_missing_sources_raises(self) -> None:
        spec = _merit_order_spec(ProfileRef(source='load', variable='demand'))
        with pytest.raises(KeyError, match='source'):
            spec.optimize()
