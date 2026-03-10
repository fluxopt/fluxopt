"""Mathematical correctness tests for carrier balance & dispatch."""

import numpy as np
from numpy.testing import assert_allclose

from fluxopt import Effect, Flow, Port

from .conftest import ts


class TestCarrierBalance:
    def test_merit_order_dispatch(self, optimize):
        """Proves: Carrier balance forces total supply = demand, and the optimizer
        dispatches sources in merit order (cheapest first, up to capacity).

        Src1: 1€/kWh, max 20. Src2: 2€/kWh, max 20. Demand=30 per timestep.
        Optimal: Src1=20, Src2=10.

        Sensitivity: If bus balance allowed oversupply, Src2 could be zero → cost=40.
        If merit order were wrong (Src2 first), cost=100. Only correct bus balance
        with merit order yields cost=80 and the exact flow split [20,10].
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect('cost', is_objective=True)],
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
        """Proves: Two nodes on the same carrier get independent balance equations.

        Node A: demand=50, cheap source @1€ (cap 100), expensive source @5€ (cap 100).
        Node B: demand=80, cheap source @1€ (cap 100), expensive source @5€ (cap 100).

        If nodes were merged, total demand=130 could be served entirely by cheap
        sources (total cap 200) → cost=2*130=260.
        With independent balances, each node uses only its own cheap source:
        cost = 2*(50*1 + 80*1) = 260 (same here, but flow split differs).

        The key proof is the flow split: node A's cheap source serves exactly 50,
        node B's cheap source serves exactly 80. No cross-node dispatch.
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect('cost', is_objective=True)],
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
        """Proves: A cheap source on node A cannot serve demand on node B.

        Node A: cheap source @1€ (cap 200). Node B: expensive source @10€ (cap 200).
        Both nodes have demand=50. If cross-subsidization were possible, the optimizer
        would route all demand through cheap_a → cost=2*100=200.
        With independent nodes: cost = 2*(50*1 + 50*10) = 1100.
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect('cost', is_objective=True)],
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
        """Proves: Single-node and multi-node carriers coexist correctly.

        Electricity: single-node carrier, shared balance.
        Heat: two nodes A and B, independent balances.

        All flows share the same optimization. Electricity demand is served by
        the single grid source. Heat demands are served by their respective node sources.
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect('cost', is_objective=True)],
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
