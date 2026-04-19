"""Mathematical correctness tests for carrier balance & dispatch."""

import numpy as np
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port

from .conftest import ts


class TestCarrierBalance:
    def test_merit_order_dispatch(self, optimize):
        """Verify merit-order dispatch yields Src1=20, Src2=10 for demand=30."""
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([30, 30])),
                    ],
                ),
                Port(
                    'Src1',
                    imports=[
                        Flow('Heat', effects_per_flow_hour={'cost': 1}, size=20),
                    ],
                ),
                Port(
                    'Src2',
                    imports=[
                        Flow('Heat', effects_per_flow_hour={'cost': 2}, size=20),
                    ],
                ),
            ],
        )
        # Src1 at max 20 @1€, Src2 covers remaining 10 @2€
        # cost = 2*(20*1 + 10*2) = 80
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 80.0, rtol=1e-5)
        # Verify individual flows to confirm dispatch split
        src1 = result.flow_rate('Src1(Heat)').values
        src2 = result.flow_rate('Src2(Heat)').values
        assert_allclose(src1, [20, 20], rtol=1e-5)
        assert_allclose(src2, [10, 10], rtol=1e-5)


class TestMultiNodeBalance:
    def test_nodes_balance_independently(self, optimize):
        """Verify two nodes on the same carrier balance independently (no cross-node dispatch)."""
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('heat', nodes=['A', 'B'])],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port('cheap_a', imports=[Flow('heat', node='A', size=100, effects_per_flow_hour={'cost': 1})]),
                Port('cheap_b', imports=[Flow('heat', node='B', size=100, effects_per_flow_hour={'cost': 1})]),
                Port('exp_a', imports=[Flow('heat', node='A', size=100, effects_per_flow_hour={'cost': 5})]),
                Port('exp_b', imports=[Flow('heat', node='B', size=100, effects_per_flow_hour={'cost': 5})]),
                Port('demand_a', exports=[Flow('heat', node='A', size=1, fixed_relative_profile=[50, 50])]),
                Port('demand_b', exports=[Flow('heat', node='B', size=1, fixed_relative_profile=[80, 80])]),
            ],
        )
        # Each node dispatches its own cheap source, expensive unused
        assert_allclose(result.flow_rate('cheap_a(heat:A)').values, [50, 50], rtol=1e-5)
        assert_allclose(result.flow_rate('cheap_b(heat:B)').values, [80, 80], rtol=1e-5)
        assert_allclose(result.flow_rate('exp_a(heat:A)').values, [0, 0], atol=1e-6)
        assert_allclose(result.flow_rate('exp_b(heat:B)').values, [0, 0], atol=1e-6)

    def test_nodes_cannot_cross_subsidize(self, optimize):
        """Verify a cheap source on node A cannot serve demand on node B."""
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('heat', nodes=['A', 'B'])],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port('cheap_a', imports=[Flow('heat', node='A', size=200, effects_per_flow_hour={'cost': 1})]),
                Port('exp_b', imports=[Flow('heat', node='B', size=200, effects_per_flow_hour={'cost': 10})]),
                Port('demand_a', exports=[Flow('heat', node='A', size=1, fixed_relative_profile=[50, 50])]),
                Port('demand_b', exports=[Flow('heat', node='B', size=1, fixed_relative_profile=[50, 50])]),
            ],
        )
        # Node B must use expensive source — can't tap into node A's cheap source
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 1100.0, rtol=1e-5)
        assert_allclose(result.flow_rate('cheap_a(heat:A)').values, [50, 50], rtol=1e-5)
        assert_allclose(result.flow_rate('exp_b(heat:B)').values, [50, 50], rtol=1e-5)

    def test_mixed_single_and_multi_node(self, optimize):
        """Verify single-node and multi-node carriers coexist in one optimization."""
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier('elec'), Carrier('heat', nodes=['A', 'B'])],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                # Electricity: single-node
                Port('grid', imports=[Flow('elec', size=200, effects_per_flow_hour={'cost': 0.1})]),
                Port('elec_demand', exports=[Flow('elec', size=1, fixed_relative_profile=[30, 30])]),
                # Heat: multi-node
                Port('boiler_a', imports=[Flow('heat', node='A', size=100, effects_per_flow_hour={'cost': 1})]),
                Port('boiler_b', imports=[Flow('heat', node='B', size=100, effects_per_flow_hour={'cost': 2})]),
                Port('bldg_a', exports=[Flow('heat', node='A', size=1, fixed_relative_profile=[40, 40])]),
                Port('bldg_b', exports=[Flow('heat', node='B', size=1, fixed_relative_profile=[60, 60])]),
            ],
        )
        # Electricity balance: grid serves 30 per timestep
        assert_allclose(result.flow_rate('grid(elec)').values, [30, 30], rtol=1e-5)
        # Heat node A: boiler_a serves 40
        assert_allclose(result.flow_rate('boiler_a(heat:A)').values, [40, 40], rtol=1e-5)
        # Heat node B: boiler_b serves 60
        assert_allclose(result.flow_rate('boiler_b(heat:B)').values, [60, 60], rtol=1e-5)
        # Total cost: 2*(30*0.1 + 40*1 + 60*2) = 2*(3+40+120) = 326
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 326.0, rtol=1e-5)
