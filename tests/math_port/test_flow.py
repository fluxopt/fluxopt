"""Mathematical correctness tests for flow constraints."""

import numpy as np
from numpy.testing import assert_allclose

from fluxopt import Carrier, Converter, Effect, Flow, Port

from .conftest import ts, waste


class TestFlowConstraints:
    def test_relative_rate_min(self, optimize):
        """Proves: relative_rate_min enforces a minimum flow rate as a fraction of size
        when the unit is active (status=1).

        Boiler (size=100, relative_rate_min=0.4). When on, must produce at least 40 kW.
        Demand=[30,30]. Since 30 < 40, boiler must produce 40 and excess is absorbed.

        Sensitivity: Without relative_rate_min, boiler produces exactly 30 each timestep
        → cost=60. With relative_rate_min=0.4, must produce 40 → cost=80.
        """

        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost')],
            objective='cost',
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([30, 30])),
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
                    thermal_flow=Flow('Heat', size=100, relative_rate_min=0.4),
                ),
            ],
        )
        # Must produce at least 40 (relative_rate_min=0.4 * size=100)
        # cost = 2 * 40 = 80 (vs 60 without the constraint)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)
        # Verify flow rate is at least 40
        flow = result.flow_rate('Boiler(Heat)').values
        assert all(f >= 40.0 - 1e-5 for f in flow), f'Flow below relative_rate_min: {flow}'

    def test_relative_rate_max(self, optimize):
        """Proves: relative_rate_max limits the maximum flow rate as a fraction of size.

        Source (size=100, relative_rate_max=0.5). Max output = 50 kW.
        Demand=[60,60]. Can only get 50 from CheapSrc, rest from ExpensiveSrc.

        Sensitivity: Without relative_rate_max, CheapSrc covers all 60 → cost=120.
        With relative_rate_max=0.5, CheapSrc capped at 50 (2*50*1=100),
        ExpensiveSrc covers 10 each timestep (2*10*5=100) → total cost=200.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost')],
            objective='cost',
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([60, 60])),
                    ],
                ),
                Port(
                    'CheapSrc',
                    imports=[
                        Flow('Heat', size=100, relative_rate_max=0.5, effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                Port(
                    'ExpensiveSrc',
                    imports=[
                        Flow('Heat', effects_per_flow_hour={'cost': 5}),
                    ],
                ),
            ],
        )
        # CheapSrc capped at 50 (relative_rate_max=0.5 * size=100): 2 * 50 * 1 = 100
        # ExpensiveSrc covers remaining 10 each timestep: 2 * 10 * 5 = 100
        # Total = 200
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 200.0, rtol=1e-5)
        # Verify CheapSrc flow rate is at most 50
        flow = result.flow_rate('CheapSrc(Heat)').values
        assert all(f <= 50.0 + 1e-5 for f in flow), f'Flow above relative_rate_max: {flow}'

    def test_flow_hours_max(self, optimize):
        """Proves: flow_hours_max limits the total cumulative flow-hours per period.

        CheapSrc (flow_hours_max=30). Total allowed = 30 kWh over horizon.
        Demand=[20,20,20] (total=60). Must split between cheap and expensive.

        Sensitivity: Without flow_hours_max, all from CheapSrc → cost=60.
        With flow_hours_max=30, CheapSrc limited to 30, ExpensiveSrc covers 30 → cost=180.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost')],
            objective='cost',
            ports=[
                Port('Demand', exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([20, 20, 20]))]),
                Port('CheapSrc', imports=[Flow('Heat', flow_hours_max=30, effects_per_flow_hour={'cost': 1})]),
                Port('ExpensiveSrc', imports=[Flow('Heat', effects_per_flow_hour={'cost': 5})]),
            ],
        )
        # CheapSrc: 30 * 1 = 30. ExpensiveSrc: 30 * 5 = 150. Total = 180.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 180.0, rtol=1e-5)
        cheap_total = result.flow_rate('CheapSrc(Heat)').values.sum()
        assert cheap_total <= 30.0 + 1e-5, f'CheapSrc above flow_hours_max: {cheap_total}'

    def test_flow_hours_min(self, optimize):
        """Proves: flow_hours_min forces a minimum total cumulative flow-hours per period.

        ExpensiveSrc (flow_hours_min=40). Must produce at least 40 kWh total.
        Demand=[30,30] (total=60). CheapSrc is preferred but ExpensiveSrc must hit 40.

        Sensitivity: Without flow_hours_min, all from CheapSrc → cost=60.
        With flow_hours_min=40, ExpensiveSrc forced to produce 40 → cost=220.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost')],
            objective='cost',
            ports=[
                Port('Demand', exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([30, 30]))]),
                Port('CheapSrc', imports=[Flow('Heat', effects_per_flow_hour={'cost': 1})]),
                Port('ExpensiveSrc', imports=[Flow('Heat', flow_hours_min=40, effects_per_flow_hour={'cost': 5})]),
            ],
        )
        # ExpensiveSrc: 40 * 5 = 200. CheapSrc: 20 * 1 = 20. Total = 220.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 220.0, rtol=1e-5)
        expensive_total = result.flow_rate('ExpensiveSrc(Heat)').values.sum()
        assert expensive_total >= 40.0 - 1e-5, f'ExpensiveSrc below flow_hours_min: {expensive_total}'

    def test_load_factor_max(self, optimize):
        """Proves: load_factor_max limits utilization to (flow_hours) / (size * total_hours).

        CheapSrc (size=50, load_factor_max=0.5). Over 2 hours, max flow_hours = 50 * 2 * 0.5 = 50.
        Demand=[40,40] (total=80). CheapSrc capped at 50 total.

        Sensitivity: Without load_factor_max, CheapSrc covers 80 → cost=80.
        With load_factor_max=0.5, CheapSrc limited to 50, ExpensiveSrc covers 30 → cost=200.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost')],
            objective='cost',
            ports=[
                Port('Demand', exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([40, 40]))]),
                Port(
                    'CheapSrc',
                    imports=[Flow('Heat', size=50, load_factor_max=0.5, effects_per_flow_hour={'cost': 1})],
                ),
                Port('ExpensiveSrc', imports=[Flow('Heat', effects_per_flow_hour={'cost': 5})]),
            ],
        )
        # CheapSrc: 50 * 1 = 50. ExpensiveSrc: 30 * 5 = 150. Total = 200.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 200.0, rtol=1e-5)
        cheap_total = result.flow_rate('CheapSrc(Heat)').values.sum()
        assert cheap_total <= 50.0 + 1e-5, f'CheapSrc above load_factor_max: {cheap_total}'

    def test_load_factor_min(self, optimize):
        """Proves: load_factor_min forces minimum utilization (flow_hours) / (size * total_hours).

        ExpensiveSrc (size=100, load_factor_min=0.3). Over 2 hours, min flow_hours = 100 * 2 * 0.3 = 60.
        Demand=[30,30] (total=60). ExpensiveSrc must produce at least 60.

        Sensitivity: Without load_factor_min, all from CheapSrc → cost=60.
        With load_factor_min=0.3, ExpensiveSrc forced to produce 60 → cost=300.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost')],
            objective='cost',
            ports=[
                Port('Demand', exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([30, 30]))]),
                Port('CheapSrc', imports=[Flow('Heat', effects_per_flow_hour={'cost': 1})]),
                Port(
                    'ExpensiveSrc',
                    imports=[Flow('Heat', size=100, load_factor_min=0.3, effects_per_flow_hour={'cost': 5})],
                ),
            ],
        )
        # ExpensiveSrc: 60 * 5 = 300. CheapSrc: 0. Total = 300.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 300.0, rtol=1e-5)
        expensive_total = result.flow_rate('ExpensiveSrc(Heat)').values.sum()
        assert expensive_total >= 60.0 - 1e-5, f'ExpensiveSrc below load_factor_min: {expensive_total}'
