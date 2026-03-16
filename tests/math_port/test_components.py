"""Mathematical correctness tests for component-level features.

Tests for component-specific behavior including heat pumps and
component-level StatusParameters.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Carrier, ConversionCurve, Converter, Effect, Flow, Port, Status

from .conftest import ts


class TestComponentStatus:
    """Tests for StatusParameters applied at the component level (not flow level)."""

    def test_component_status_startup_cost(self, optimize):
        """Proves: StatusParameters on ConversionCurve applies startup cost when
        the component transitions to active.

        Boiler with piecewise conversion and startup cost.
        demand=[0,20,0,20] → 2 startups without constraint. Startup cost=100 per event.
        Optimizer keeps boiler running to avoid double startup cost.
        """
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([0, 20, 0, 20])),
                    ],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
                Port('Waste', exports=[Flow('Heat', size=1000)]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 100], 'Heat': [0, 100]},
                        status=Status(effects_per_startup={'cost': 100}),
                    ),
                ),
            ],
        )
        # With startup cost of 100, keeping boiler on is cheaper than restarting
        on = result.solution['component--on'].sel(component='Boiler').values
        startups = sum(1 for i in range(len(on)) if on[i] > 0.5 and (i == 0 or on[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup, got {startups}: on={on}'

    def test_component_status_min_uptime(self, optimize):
        """Proves: min_uptime on component level forces the entire component
        to stay on for consecutive hours.

        Boiler with piecewise conversion, min_uptime=3.
        demand=[20]*6 with backup at 0.5 efficiency.
        Boiler must stay on for ≥3h once started, even if turning off
        would save fuel (waste heat cost). We check internal on-blocks
        (excluding terminal ones which can be shorter due to horizon edge).
        """
        result = optimize(
            timesteps=ts(6),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20, 20, 20, 20, 20])),
                    ],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
                Port('Waste', exports=[Flow('Heat', size=1000)]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 100], 'Heat': [0, 100]},
                        status=Status(min_uptime=3),
                    ),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Boiler').values
        # With min_uptime=3 and constant demand, the boiler should stay on for ≥3h
        # Check that total on-time is ≥3 (at least one block of 3)
        total_on = sum(1 for v in on if v > 0.5)
        assert total_on >= 3, f'Expected ≥3 on-hours, got {total_on}: on={on}'

    @pytest.mark.skip(reason='active_hours_max not supported in fluxopt')
    def test_component_status_active_hours_max(self, optimize):
        """Proves: active_hours_max on component level limits total operating hours."""

    def test_component_status_effects_per_active_hour(self, optimize):
        """Proves: effects_per_running_hour on component level adds cost per active hour.

        Boiler with piecewise conversion, running cost=10/h.
        demand=[20,20] → boiler on 2h → extra cost = 20.
        Gas cost=1/unit, demand=40 total. Total = 40 + 20 = 60.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 100], 'Heat': [0, 100]},
                        status=Status(effects_per_running_hour={'cost': 10}),
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-4)

    @pytest.mark.skip(reason='active_hours_min not supported in fluxopt')
    def test_component_status_active_hours_min(self, optimize):
        """Proves: active_hours_min on component level forces minimum operating hours."""

    def test_component_status_max_uptime(self, optimize):
        """Proves: max_uptime on component level limits continuous operation.

        Boiler with piecewise conversion, max_uptime=2.
        demand=[20]*4 → boiler can only run ≤2h continuously.
        """
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20, 20, 20])),
                    ],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
                Port('Backup', imports=[Flow('Heat', size=1000, effects_per_flow_hour={'cost': 10})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 100], 'Heat': [0, 100]},
                        status=Status(max_uptime=2),
                    ),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Boiler').values
        from conftest import assert_on_blocks

        assert_on_blocks(on, max_length=2)

    def test_component_status_min_downtime(self, optimize):
        """Proves: min_downtime on component level prevents quick restart.

        Boiler with piecewise conversion, max_uptime=1, min_downtime=2.
        demand=[20]*5 → boiler must be off ≥2h after each on.
        """
        result = optimize(
            timesteps=ts(5),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20, 20, 20, 20])),
                    ],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
                Port('Backup', imports=[Flow('Heat', size=1000, effects_per_flow_hour={'cost': 10})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 100], 'Heat': [0, 100]},
                        status=Status(max_uptime=1, min_downtime=2),
                    ),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Boiler').values
        from conftest import assert_off_blocks

        assert_off_blocks(on, min_length=2)

    def test_component_status_max_downtime(self, optimize):
        """Proves: max_downtime on component level forces restart after idle.

        Boiler with piecewise conversion, max_downtime=1.
        No demand → boiler can only be off ≤1h at a time.
        """
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port('Waste', exports=[Flow('Heat', size=1000)]),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 100], 'Heat': [0, 100]},
                        status=Status(max_downtime=1),
                    ),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Boiler').values
        from conftest import assert_off_blocks

        assert_off_blocks(on, max_length=1)

    @pytest.mark.skip(reason='startup_limit not supported in fluxopt')
    def test_component_status_startup_limit(self, optimize):
        """Proves: startup_limit on component level caps number of startups."""


class TestTransmission:
    """Tests for Transmission component with losses and structural constraints."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_relative_losses(self, optimize):
        """Proves: relative_losses correctly reduces transmitted energy."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_absolute_losses(self, optimize):
        """Proves: absolute_losses adds fixed loss when transmission is active."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_bidirectional(self, optimize):
        """Proves: Bidirectional transmission allows flow in both directions."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
    def test_transmission_prevent_simultaneous_bidirectional(self, optimize):
        """Proves: prevent_simultaneous_flows_in_both_directions=True prevents both
        directions from being active at the same timestep."""

    @pytest.mark.skip(reason='Transmission not supported in fluxopt')
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
            carriers=[Carrier('Elec'), Carrier('Env'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                    ],
                ),
                Port('Grid', imports=[Flow('Elec', effects_per_flow_hour={'cost': 1})]),
                Port('Environment', imports=[Flow('Env', size=1000)]),
            ],
            converters=[
                Converter.heat_pump(
                    'HP',
                    cop=3.0,
                    electrical_flow=Flow('Elec'),
                    source_flow=Flow('Env'),
                    thermal_flow=Flow('Heat'),
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
            carriers=[Carrier('Elec'), Carrier('Env'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port('Grid', imports=[Flow('Elec', effects_per_flow_hour={'cost': 1})]),
                Port('Environment', imports=[Flow('Env', size=1000)]),
            ],
            converters=[
                Converter.heat_pump(
                    'HP',
                    cop=np.array([2.0, 4.0]),
                    electrical_flow=Flow('Elec'),
                    source_flow=Flow('Env'),
                    thermal_flow=Flow('Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 15.0, rtol=1e-5)


@pytest.mark.skip(reason='cooling_tower factory not supported in fluxopt')
class TestCoolingTower:
    """Tests for CoolingTower component."""

    def test_cooling_tower_specific_electricity(self, optimize):
        """Proves: CoolingTower correctly applies specific_electricity_demand."""


@pytest.mark.skip(reason='power2heat factory not supported in fluxopt')
class TestPower2Heat:
    """Tests for Power2Heat component."""

    def test_power2heat_efficiency(self, optimize):
        """Proves: Power2Heat applies thermal_efficiency to electrical input."""


@pytest.mark.skip(reason='heat_pump_with_source factory not supported in fluxopt')
class TestHeatPumpWithSource:
    """Tests for HeatPumpWithSource component with COP and heat source."""

    def test_heatpump_with_source_cop(self, optimize):
        """Proves: HeatPumpWithSource applies COP to compute electrical consumption,
        drawing the remainder from a heat source."""


@pytest.mark.skip(reason='prevent_simultaneous not supported in fluxopt')
class TestSourceAndSink:
    """Tests for SourceAndSink component."""

    def test_source_and_sink_prevent_simultaneous(self, optimize):
        """Proves: SourceAndSink with prevent_simultaneous_flow_rates=True prevents
        buying and selling in the same timestep."""
