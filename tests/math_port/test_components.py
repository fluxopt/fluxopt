"""Mathematical correctness tests for component-level features.

Tests for component-specific behavior including heat pumps, power-to-heat and
grid buy/sell exclusion. Transmission is not yet supported.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Carrier, Converter, Effect, Flow, Port, Storage

from .conftest import ts


class TestTransmission:
    """Tests for Transmission component with losses and structural constraints."""

    @pytest.mark.skip(reason='Transmission not supported — issue #202')
    def test_transmission_relative_losses(self, optimize):
        """Proves: relative_losses correctly reduces transmitted energy."""

    @pytest.mark.skip(reason='Transmission not supported — issue #202')
    def test_transmission_absolute_losses(self, optimize):
        """Proves: absolute_losses adds fixed loss when transmission is active."""

    @pytest.mark.skip(reason='Transmission not supported — issue #202')
    def test_transmission_bidirectional(self, optimize):
        """Proves: Bidirectional transmission allows flow in both directions."""

    @pytest.mark.skip(reason='Transmission not supported — issue #202')
    def test_transmission_prevent_simultaneous_bidirectional(self, optimize):
        """Proves: prevent_simultaneous_flows_in_both_directions=True prevents both
        directions from being active at the same timestep."""

    @pytest.mark.skip(reason='Transmission not supported — issue #202')
    def test_transmission_status_startup_cost(self, optimize):
        """Proves: StatusParameters on Transmission applies startup cost."""


class TestHeatPump:
    """Tests for HeatPump component with COP and source heat."""

    def test_heatpump_cop(self, optimize):
        """Proves: HeatPump splits demand into electrical and source via COP.

        demand=30/t, COP=3 → elec=10/t, source=20/t → cost=20.
        Sensitivity: Without source flow (COP=1 effective), cost=60.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec'), Carrier(id='Env'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                    ],
                ),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
                Port(id='Environment', imports=[Flow(carrier='Env', size=1000)]),
            ],
            converters=[
                Converter.heat_pump(
                    'HP',
                    cop=3.0,
                    electrical_flow=Flow(carrier='Elec'),
                    source_flow=Flow(carrier='Env'),
                    thermal_flow=Flow(carrier='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)

    def test_heatpump_variable_cop(self, optimize):
        """Proves: HeatPump accepts time-varying COP array.

        demand=20/t, COP=[2,4] → elec=[10,5]=15, source=[10,15] → cost=15.
        Sensitivity: If scalar cop=3 used, elec=13.33. Only time-varying gives 15.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec'), Carrier(id='Env'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
                Port(id='Environment', imports=[Flow(carrier='Env', size=1000)]),
            ],
            converters=[
                Converter.heat_pump(
                    'HP',
                    cop=np.array([2.0, 4.0]),
                    electrical_flow=Flow(carrier='Elec'),
                    source_flow=Flow(carrier='Env'),
                    thermal_flow=Flow(carrier='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 15.0, rtol=1e-5)


@pytest.mark.skip(reason='cooling_tower factory not implemented — issue #252')
class TestCoolingTower:
    """Tests for CoolingTower component."""

    def test_cooling_tower_specific_electricity(self, optimize):
        """Proves: CoolingTower correctly applies specific_electricity_demand.

        specific_electricity_demand=0.1 (kWel/kWth): for 200 kWth rejected,
        needs 20 kWel → cost=20. Expressible today as an input-only Converter
        (inputs=[thermal, elec], outputs=[],
        conversion_factors=[{'Elec': 1, 'Heat': -0.1}]); only the factory is missing.
        """


class TestPower2Heat:
    """Tests for Power2Heat component."""

    def test_power2heat_efficiency(self, optimize):
        """Proves: Power2Heat applies efficiency to electrical input.

        efficiency=0.9. Demand=40 heat over 2 timesteps.
        Elec needed = 40 / 0.9 ≈ 44.44 → cost≈44.44.

        Sensitivity: If efficiency ignored (=1), elec=40 → cost=40.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter.power2heat(
                    'P2H',
                    efficiency=0.9,
                    electrical_flow=Flow(carrier='Elec'),
                    thermal_flow=Flow(carrier='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0 / 0.9, rtol=1e-5)


class TestHeatPumpWithSource:
    """Tests for heat pump with explicit source-heat balance."""

    def test_heatpump_with_source_cop(self, optimize):
        """Proves: heat_pump applies COP to compute electrical consumption,
        drawing the remainder from a heat source.

        cop=3. Demand=60 heat over 2 timesteps.
        Elec = 60/3 = 20 → cost=20. Heat source provides 60 - 20 = 40.

        Sensitivity: If cop=1, elec=60 → cost=60. With cop=3, cost=20.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec'), Carrier(id='SourceHeat'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                    ],
                ),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
                Port(id='FreeHeat', imports=[Flow(carrier='SourceHeat')]),
            ],
            converters=[
                Converter.heat_pump(
                    'HP',
                    cop=3.0,
                    electrical_flow=Flow(carrier='Elec'),
                    source_flow=Flow(carrier='SourceHeat', short_id='source'),
                    thermal_flow=Flow(carrier='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)
        assert_allclose(result.flow_rate('HP(source)').values, [20, 20], rtol=1e-5)


class TestSourceAndSink:
    """Tests for grid buy/sell with mutual exclusion (flixopt SourceAndSink)."""

    def test_source_and_sink_prevent_simultaneous(self, optimize):
        """Proves: buying and selling are mutually exclusive per timestep.

        flixopt modeled this as SourceAndSink(prevent_simultaneous_flow_rates=True).
        fluxopt has no Port-level exclusion, so the grid connection is modeled as a
        Storage with an unconstrained level (large capacity, cyclic=False) whose
        prevent_simultaneous binary provides the same exclusion:
        discharging = buy, charging = sell.

        Solar=[30, 30, 0]. Demand=[10, 10, 10]. Buy @5€, sell @-1€.
        t0,t1: excess 20 → sell 20 (revenue 20 each = -40). t2: deficit 10 → buy 10 (50).

        Sensitivity: Cost = 50 - 40 = 10.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Solar',
                    imports=[
                        Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([30, 30, 0])),
                    ],
                ),
            ],
            storages=[
                Storage(
                    id='GridConnection',
                    charging=Flow(carrier='Elec', short_id='sell', size=100, effects_per_flow_hour={'cost': -1}),
                    discharging=Flow(carrier='Elec', short_id='buy', size=100, effects_per_flow_hour={'cost': 5}),
                    capacity=1000,
                    cyclic=False,
                    prevent_simultaneous=True,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-5)
        buy = result.flow_rate('GridConnection(buy)').values
        sell = result.flow_rate('GridConnection(sell)').values
        assert not ((buy > 1e-5) & (sell > 1e-5)).any(), f'Simultaneous buy/sell: buy={buy}, sell={sell}'
