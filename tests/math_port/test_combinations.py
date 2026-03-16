"""Mathematical correctness tests for COMBINATIONS of features.

These tests verify that status parameters, investment sizing, and effects
work correctly when combined — catching interaction bugs that single-feature
tests miss.
"""

import numpy as np
import pytest
from conftest import assert_off_blocks, assert_on_blocks
from numpy.testing import assert_allclose

from fluxopt import (
    Carrier,
    ConversionCurve,
    Converter,
    Effect,
    Flow,
    PiecewiseInvestment,
    Port,
    Sizing,
    Status,
)

from .conftest import ts, waste


class TestPiecewiseWithInvestment:
    """Tests combining PiecewiseConversion with PiecewiseInvestment."""

    def test_piecewise_conversion_with_investment_sizing(self, optimize):
        """Proves: PiecewiseInvestment on a piecewise converter gates operation by period.

        Boiler with PiecewiseInvestment(mandatory=True) must be built in one period.
        periods=[2025, 2030], demand=30 both periods.
        Investment cost = 10/MW. Size = max_bp = 100.
        Fuel cost = 1/MWh. At demand=30 → Gas=33.3 (eta≈0.9 in segment 1).

        Total = invest_cost(100*10=1000) + fuel(33.3*2=66.7).
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([30.0]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 50, 100], 'Heat': [0, 45, 80]},
                        size=PiecewiseInvestment(
                            mandatory=True,
                            effects_per_size={'cost': 10},
                        ),
                    ),
                ),
            ],
            periods=[2025, 2030],
        )
        # Investment cost: 100 * 10 = 1000 (one-time, charged once)
        # Fuel per period: demand=30 → Gas≈33.3 (interpolated), cost≈33.3
        # Each period weighted 5 years: temporal cost = 33.3 * 5 * 2 = ~333
        # Total: 1000 + 333 ≈ 1333
        total = result.effect_totals.sel(effect='cost').values
        assert total.sum() > 1000, f'Expected total > 1000 (includes invest), got {total.sum()}'
        # Verify boiler is active in both periods (mandatory build)
        pw_active = result.solution['pw_invest--active'].sel(pw_converter='Boiler')
        assert pw_active.sum() >= 2, f'Expected active in both periods, got {pw_active.values}'

    def test_piecewise_invest_cost_with_optional_skip(self, optimize):
        """Proves: Optional PiecewiseInvestment (mandatory=False) allows skipping build.

        High invest cost (10000/MW) with cheap backup.
        Optimizer should skip building the boiler entirely.
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([30.0]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
                Port('Backup', imports=[Flow('Heat', size=1000, effects_per_flow_hour={'cost': 0.5})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 50, 100], 'Heat': [0, 45, 80]},
                        size=PiecewiseInvestment(
                            mandatory=False,
                            effects_per_size={'cost': 10000},
                        ),
                    ),
                ),
            ],
            periods=[2025, 2030],
        )
        # Backup at 0.5/unit * 30 units = 15 per period << invest cost
        pw_active = result.solution['pw_invest--active'].sel(pw_converter='Boiler')
        assert pw_active.sum() == 0, f'Expected no build (too expensive), got active={pw_active.values}'


class TestPiecewiseWithStatus:
    """Tests combining PiecewiseConversion with StatusParameters."""

    def test_piecewise_nonlinear_conversion_with_startup_cost(self, optimize):
        """Proves: PiecewiseConversion and startup costs interact correctly.

        Boiler with nonlinear efficiency and startup cost=50.
        demand=[0,30,0,30] → 2 startups = 100. Keeping on costs less (run-waste).
        """
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([0, 30, 0, 30]))],
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
                        breakpoints={'Gas': [0, 50, 100], 'Heat': [0, 45, 80]},
                        status=Status(effects_per_startup={'cost': 50}),
                    ),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Boiler').values
        startups = sum(1 for i in range(len(on)) if on[i] > 0.5 and (i == 0 or on[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup: on={on}'

    def test_piecewise_minimum_load_with_status(self, optimize):
        """Proves: Piecewise gap enforces minimum load, interacting with status.

        breakpoints: Gas=[0, 20, 100], Heat=[0, 15, 80]
        With status → can be off (Gas=Heat=0) or on with Gas≥20.
        demand=5 → boiler at minimum (Gas=20, Heat=15) or off + backup.
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([5.0]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
                Port('Backup', imports=[Flow('Heat', size=1000, effects_per_flow_hour={'cost': 0.5})]),
                Port('Waste', exports=[Flow('Heat', size=1000)]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 20, 100], 'Heat': [0, 15, 80]},
                        status=Status(),
                    ),
                ),
            ],
        )
        # Backup at 0.5/unit for 5 units = 2.5 is cheaper than boiler at Gas=20 → cost=20
        backup_rate = result.flow_rates.sel(flow='Backup(Heat)').values[0]
        assert_allclose(backup_rate, 5.0, atol=1e-4)

    def test_piecewise_no_zero_point_with_status(self, optimize):
        """Proves: Piecewise WITHOUT off-state piece interacts with StatusParameters.

        breakpoints start at Gas=20 (no zero point), so minimum load = 20.
        With status, the component can still be turned off (on=0 → all flows=0).
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([0, 30]))],
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
                        breakpoints={'Gas': [20, 50, 100], 'Heat': [15, 45, 80]},
                        status=Status(),
                    ),
                ),
            ],
        )
        # At t=0: demand=0, boiler off → Gas=0
        gas_t0 = result.flow_rates.sel(flow='Boiler(Gas)').values[0]
        assert_allclose(gas_t0, 0.0, atol=1e-4)

    def test_piecewise_no_zero_point_startup_cost(self, optimize):
        """Proves: Piecewise without zero point + startup cost work together.

        breakpoints start at Gas=20, startup cost=100.
        demand=[0,30,0,30] → 2 startups cost 200.
        Staying on costs less fuel waste.
        """
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([0, 30, 0, 30]))],
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
                        breakpoints={'Gas': [20, 50, 100], 'Heat': [15, 45, 80]},
                        status=Status(effects_per_startup={'cost': 100}),
                    ),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Boiler').values
        startups = sum(1 for i in range(len(on)) if on[i] > 0.5 and (i == 0 or on[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup: on={on}'


class TestPiecewiseThreeSegments:
    """Tests for piecewise conversion with 3+ segments."""

    def test_three_segment_piecewise(self, optimize):
        """Proves: 3-segment PiecewiseConversion correctly selects the optimal segment.

        breakpoints: Gas=[0, 30, 60, 100], Heat=[0, 27, 48, 70]
        Segment 1: eta=0.9, Segment 2: eta=0.7, Segment 3: eta=0.55

        demand=27 → BP1 (Gas=30). demand=48 → BP2 (Gas=60).
        Total cost = 30 + 60 = 90.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([27, 48]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 30, 60, 100], 'Heat': [0, 27, 48, 70]},
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 90.0, rtol=1e-4)

    def test_three_segment_low_load_selection(self, optimize):
        """Proves: With 3 segments, low demand correctly uses segment 1.

        breakpoints: Gas=[0, 30, 60, 100], Heat=[0, 27, 48, 70]
        demand=13.5 → interpolate in segment 1 (Gas=15, Heat=13.5).
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([13.5]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 30, 60, 100], 'Heat': [0, 27, 48, 70]},
                    ),
                ),
            ],
        )
        # Heat=13.5 is exactly half of segment 1 → Gas = 0 + 0.5*(30-0) = 15
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 15.0, rtol=1e-4)

    def test_three_segment_mid_load_selection(self, optimize):
        """Proves: With 3 segments, mid demand correctly uses segment 2.

        breakpoints: Gas=[0, 30, 60, 100], Heat=[0, 27, 48, 70]
        demand=37.5 → interpolate in segment 2. Heat goes 27→48 (delta=21) over Gas 30→60.
        fraction = (37.5-27)/21 = 0.5 → Gas = 30 + 0.5*30 = 45.
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([37.5]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 30, 60, 100], 'Heat': [0, 27, 48, 70]},
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 45.0, rtol=1e-4)


class TestStatusWithEffects:
    """Tests for StatusParameters contributing to non-standard effects."""

    def test_startup_cost_on_co2_effect(self, optimize):
        """Proves: effects_per_startup can contribute to a non-cost effect (CO2),
        and that this correctly interacts with effect constraints.

        CO2 capped at maximum_total=60. Boiler startup emits 50kg CO2.
        Demand=[0,20,0,20] → 2 startups = 100kg CO2. Exceeds cap!
        Optimizer must reduce startups by keeping boiler running continuously.

        Sensitivity: Without CO2 cap, 2 startups optimal. With cap=60, forced to 1 startup.
        """

        result = optimize(
            timesteps=ts(4),
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', maximum_total=60),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([0, 20, 0, 20])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow('Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                waste('Heat'),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        'Heat',
                        size=100,
                        relative_minimum=0.1,
                        prior_rates=[0],
                        status=Status(effects_per_startup={'CO2': 50}),
                    ),
                ),
            ],
            carriers=[Carrier('Gas'), Carrier('Heat')],
        )
        assert result.effect_totals.sel(effect='CO2').item() <= 60.0 + 1e-5
        # Verify only 1 startup (continuous operation)
        on = result.solution['flow--on'].sel(flow='Boiler(Heat)').values
        startups = sum(1 for i in range(len(on)) if on[i] > 0.5 and (i == 0 or on[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup, got {startups}: on={on}'

    def test_effects_per_active_hour_on_multiple_effects(self, optimize):
        """Proves: effects_per_running_hour can contribute to multiple effects simultaneously.

        Boiler with effects_per_running_hour={'cost': 10, 'CO2': 5}.
        Demand=[20,20]. Boiler on 2 hours.

        Sensitivity: Without effects_per_running_hour, costs=40, CO2=0.
        With it, costs = 40 + 2*10 = 60, CO2 = 2*5 = 10.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2'),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow('Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        'Heat',
                        size=100,
                        relative_minimum=0.1,
                        status=Status(effects_per_running_hour={'cost': 10, 'CO2': 5}),
                    ),
                ),
            ],
            carriers=[Carrier('Gas'), Carrier('Heat')],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 10.0, rtol=1e-5)


class TestInvestWithRelativeMinimum:
    """Tests combining Sizing with relative_minimum."""

    def test_invest_sizing_respects_relative_minimum(self, optimize):
        """Proves: relative_minimum on an invested flow forces the boiler OFF at
        low-demand timesteps, requiring expensive backup.

        Sensitivity: Without relative_minimum: size=50, ON both hours, fuel=55, total=80.
        With it: size=50, OFF at t=0, backup=5*10=50, total=125.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([5, 50])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow('Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                Port(
                    'Backup',
                    imports=[
                        Flow('Heat', effects_per_flow_hour={'cost': 10}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        'Heat',
                        relative_minimum=0.5,
                        size=Sizing(
                            min_size=0,
                            max_size=100,
                            mandatory=True,
                            effects_per_size={'cost': 0.5},
                        ),
                        status=Status(),
                    ),
                ),
            ],
            carriers=[Carrier('Gas'), Carrier('Heat')],
        )
        assert_allclose(result.sizes.sel(flow='Boiler(Heat)').item(), 50.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 125.0, rtol=1e-4)
        # Verify boiler is OFF at t=0
        assert result.solution['flow--on'].sel(flow='Boiler(Heat)').values[0] < 0.5


class TestConversionWithTimeVaryingEffects:
    """Tests for conversion factors with time-varying effects."""

    def test_time_varying_effects_per_flow_hour(self, optimize):
        """Proves: Time-varying effects_per_flow_hour correctly applies different rates
        per timestep when combined with conversion.

        Boiler eta=0.5. Gas cost = [1, 3]. Demand=[20, 10].
        t=0: fuel=40, cost=40. t=1: fuel=20, cost=60. Total=100.

        Sensitivity: If mean(2) were used: cost=120. Only per-timestep gives 100.
        """

        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 10])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow('Gas', effects_per_flow_hour={'cost': np.array([1, 3])}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow('Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_effects_per_flow_hour_with_dual_output_conversion(self, optimize):
        """Proves: effects_per_flow_hour applied to individual flows of a multi-output
        converter correctly accumulates effects for each flow independently.

        CHP: fuel→heat+elec. Fuel costs 1€/kWh, elec earns -2€/kWh.
        CO2: fuel emits 0.5 kg/kWh, elec avoids -0.3 kg/kWh.
        Demand=50 heat per timestep.

        Sensitivity: Total uniquely determined by conversion factors + effects.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2'),
            ],
            ports=[
                Port(
                    'HeatDemand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([50, 50])),
                    ],
                ),
                Port(
                    'ElecGrid',
                    exports=[
                        Flow('Elec', effects_per_flow_hour={'cost': -2, 'CO2': -0.3}),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow('Gas', effects_per_flow_hour={'cost': 1, 'CO2': 0.5}),
                    ],
                ),
            ],
            converters=[
                Converter.chp(
                    'CHP',
                    eta_th=0.5,
                    eta_el=0.4,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow('Heat'),
                    electrical_flow=Flow('Elec'),
                ),
            ],
            carriers=[Carrier('Elec'), Carrier('Gas'), Carrier('Heat')],
        )
        # Per ts: fuel=100, elec=40. costs: 100-80=20. CO2: 50-12=38. Total: costs=40, CO2=76.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 76.0, rtol=1e-5)


class TestPiecewiseInvestWithStatus:
    """Tests combining piecewise investment costs with status parameters."""

    def test_piecewise_invest_with_startup_cost(self, optimize):
        """Proves: PiecewiseInvestment + Status work together.

        Boiler must be built (mandatory) AND turned on to produce.
        Startup cost = 50. Investment cost = 1/MW. demand=[0,30,0,30].
        Optimizer should prefer staying on to avoid multiple startups.
        """
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([0, 30, 0, 30]))],
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
                        breakpoints={'Gas': [0, 50, 100], 'Heat': [0, 45, 80]},
                        size=PiecewiseInvestment(
                            mandatory=True,
                            effects_per_size={'cost': 1},
                        ),
                        status=Status(effects_per_startup={'cost': 50}),
                    ),
                ),
            ],
            periods=[2025, 2030],
        )
        # Check per-period: use first period (both have same timesteps)
        on = result.solution['component--on'].sel(component='Boiler', period=2025).values
        startups = sum(1 for i in range(len(on)) if on[i] > 0.5 and (i == 0 or on[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup: on={on}'


class TestStatusWithMultipleConstraints:
    """Tests combining multiple status parameters on the same flow."""

    @pytest.mark.skip(reason='startup_limit not supported in fluxopt')
    def test_startup_limit_with_max_downtime(self, optimize):
        """Proves: startup_limit and max_downtime interact correctly."""

    def test_min_uptime_with_min_downtime(self, optimize):
        """Proves: min_uptime and min_downtime together force a regular on/off pattern.

        Boiler: min_uptime=2, min_downtime=2, prior_rates=[0].
        Demand=[20]*6. Backup at eta=0.5.

        Sensitivity: Without these constraints, boiler could run all 6 hours.
        With constraints, forced into block pattern → backup needed for off blocks.
        """

        result = optimize(
            timesteps=ts(6),
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20, 20, 20, 20, 20])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow('Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        'Heat',
                        size=100,
                        relative_minimum=0.1,
                        prior_rates=[0],
                        status=Status(min_uptime=2, min_downtime=2),
                    ),
                ),
                Converter.boiler(
                    'Backup',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow('Heat', size=100),
                ),
            ],
            carriers=[Carrier('Gas'), Carrier('Heat')],
        )
        on = result.solution['flow--on'].sel(flow='CheapBoiler(Heat)').values

        # Verify min_uptime: each on-block is ≥2 hours
        assert_on_blocks(on, min_length=2)

        # Verify min_downtime: each off-block is ≥2 hours (within horizon)
        assert_off_blocks(on, min_length=2)

        # Pattern [off,on,on,on,on,on]: CheapBoiler 5h=100, Backup 1h*20/0.5=40. Total=140.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 140.0, rtol=1e-5)


class TestEffectsWithConversion:
    """Tests for effects interacting with conversion and other constraints."""

    @pytest.mark.skip(reason='share_from_periodic not supported in fluxopt')
    def test_effect_share_with_investment(self, optimize):
        """Proves: share_from_periodic works correctly with investment costs."""

    def test_effect_maximum_with_status_contribution(self, optimize):
        """Proves: Effect maximum_total correctly accounts for contributions from
        StatusParameters (effects_per_startup) when constraining.

        CO2 has maximum_total=20. Boiler startup emits 15 kg CO2.
        Fuel emits 0.1 kg CO2/kWh. Demand=[0,10,0,10].
        2 startups = 30 kg CO2 (exceeds cap). Forced to 1 startup.

        Sensitivity: Without CO2 cap, 2 startups → CO2=30+fuel. With cap=20, ≤1 startup.
        """

        result = optimize(
            timesteps=ts(4),
            effects=[
                Effect('cost', is_objective=True),
                Effect('CO2', maximum_total=20),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([0, 10, 0, 10])),
                    ],
                ),
                Port(
                    'GasSrc',
                    imports=[
                        Flow('Gas', effects_per_flow_hour={'cost': 1, 'CO2': 0.1}),
                    ],
                ),
                waste('Heat'),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow('Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        'Heat',
                        size=100,
                        relative_minimum=0.1,
                        prior_rates=[0],
                        status=Status(effects_per_startup={'CO2': 15}),
                    ),
                ),
            ],
            carriers=[Carrier('Gas'), Carrier('Heat')],
        )
        assert result.effect_totals.sel(effect='CO2').item() <= 20.0 + 1e-5


class TestInvestWithEffects:
    """Tests combining investment with effect constraints."""

    @pytest.mark.skip(reason='maximum_periodic not supported in fluxopt')
    def test_invest_per_size_on_non_cost_effect(self, optimize):
        """Proves: effects_per_size can contribute to a non-cost effect."""
