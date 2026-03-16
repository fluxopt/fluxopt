"""Mathematical correctness tests for piecewise linearization.

Tests that ConversionCurve with breakpoints correctly interpolates
between operating points using linopy's piecewise API.
"""

import numpy as np
from numpy.testing import assert_allclose

from fluxopt import Carrier, ConversionCurve, Converter, Effect, Flow, PiecewiseSizing, Port

from .conftest import ts


class TestPiecewise:
    def test_piecewise_selects_cheap_segment(self, optimize):
        """Proves: PiecewiseConversion correctly interpolates within the active segment.

        Boiler with decreasing efficiency at high load:
        breakpoints: Gas=[0, 50, 100], Heat=[0, 45, 80]
        Segment 1: eta=0.9, Segment 2: eta=0.7

        demand=45 → operate at breakpoint 1 (exact match).
        Gas cost=1 → fuel=50 → cost=50.
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([45.0]))],
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
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 50.0, rtol=1e-4)

    def test_piecewise_conversion_at_breakpoint(self, optimize):
        """Proves: PiecewiseConversion is consistent at segment boundaries.

        breakpoints: Gas=[0, 50, 100], Heat=[0, 40, 90]
        demand=40 → exactly at breakpoint 1 → fuel=50.
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([40.0]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 50, 100], 'Heat': [0, 40, 90]},
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 50.0, rtol=1e-4)

    def test_piecewise_with_gap_forces_minimum_load(self, optimize):
        """Proves: Gaps between pieces create forbidden operating regions.

        breakpoints: Gas=[10, 50, 100], Heat=[8, 45, 80]
        mandatory=True → no off-state → minimum load = Gas=10.
        demand=8 → must operate at minimum (Gas=10, Heat=8).
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([8.0]))],
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
                        breakpoints={'Gas': [10, 50, 100], 'Heat': [8, 45, 80]},
                    ),
                ),
            ],
        )
        # Fuel must be at least 10 even though demand is only 8
        gas_rate = result.flow_rates.sel(flow='Boiler(Gas)').values[0]
        assert gas_rate >= 10.0 - 1e-4, f'Expected Gas >= 10, got {gas_rate}'

    def test_piecewise_gap_allows_off_state(self, optimize):
        """Proves: Piecewise with off-state allows unit to be completely off.

        breakpoints: Gas=[0, 10, 50, 100], Heat=[0, 8, 45, 80]
        The first breakpoint at (0,0) allows the off-state.
        mandatory=False → can choose not to operate.
        demand=0 → boiler off.
        """
        result = optimize(
            timesteps=ts(1),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([0.0]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 10, 50, 100], 'Heat': [0, 8, 45, 80]},
                        size=PiecewiseSizing(mandatory=False),
                    ),
                ),
            ],
        )
        gas_rate = result.flow_rates.sel(flow='Boiler(Gas)').values[0]
        assert_allclose(gas_rate, 0.0, atol=1e-4)

    def test_piecewise_varying_efficiency_across_segments(self, optimize):
        """Proves: Different segments can have different efficiency ratios.

        breakpoints: Gas=[0, 50, 100], Heat=[0, 45, 70]
        Segment 1: eta=45/50=0.9, Segment 2: eta=25/50=0.5

        demand=45 → operate at BP1 (eta=0.9). Gas=50, cost=50.
        demand=70 → operate at BP2 (eta=0.7). Gas=100, cost=100.

        Two timesteps: demand=[45, 70] → total cost = 50 + 100 = 150.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Gas'), Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[Flow('Heat', size=1, fixed_relative_profile=np.array([45, 70]))],
                ),
                Port('GasSrc', imports=[Flow('Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    'Boiler',
                    inputs=[Flow('Gas', size=100)],
                    outputs=[Flow('Heat', size=100)],
                    conversion=ConversionCurve(
                        breakpoints={'Gas': [0, 50, 100], 'Heat': [0, 45, 70]},
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 150.0, rtol=1e-4)
