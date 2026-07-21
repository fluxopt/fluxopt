from __future__ import annotations

import pytest
from conftest import ts
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port, Sizing, optimize


class TestEffects:
    def test_single_cost_effect(self):
        """Total cost = sum(rate * coeff * dt)."""
        demand = [50.0, 80.0, 60.0]

        sink_flow = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        source_flow = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source_flow]), Port(id='demand', exports=[sink_flow])],
        )

        expected = sum(d * 0.04 for d in demand)
        assert result.objective == pytest.approx(expected, abs=1e-6)

    def test_multiple_effects(self):
        """Track cost and CO2 simultaneously, minimize cost."""

        sink_flow = Flow(
            carrier='elec',
            size=100,
            fixed_relative_profile=[0.5, 0.8, 0.6],
        )
        source_flow = Flow(
            carrier='elec',
            size=200,
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost'), Effect(id='co2', unit='kg')],
            objective='cost',
            ports=[Port(id='grid', imports=[source_flow]), Port(id='demand', exports=[sink_flow])],
        )

        demand_total = 50 + 80 + 60
        expected_cost = demand_total * 0.04
        expected_co2 = demand_total * 0.5

        assert result.objective == pytest.approx(expected_cost, abs=1e-6)
        co2_total = float(result.effect_totals.sel(effect='co2').values)
        assert co2_total == pytest.approx(expected_co2, abs=1e-6)

    def test_effect_maximum(self):
        """Effect max_total constraint limits total emissions."""

        sink_flow = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        # Two sources with different cost/co2 tradeoffs
        cheap_dirty = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
        expensive_clean = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})

        co2_limit = 100.0  # demand_total = 190, so can't use all cheap
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost'), Effect(id='co2', total_max=co2_limit)],
            objective='cost',
            ports=[
                Port(id='cheap_src', imports=[cheap_dirty]),
                Port(id='clean_src', imports=[expensive_clean]),
                Port(id='demand', exports=[sink_flow]),
            ],
        )

        co2_total = float(result.effect_totals.sel(effect='co2').values)
        assert co2_total <= co2_limit + 1e-6

    def test_time_varying_cost(self):
        """Time-varying costs are tracked correctly."""
        prices = [0.02, 0.08, 0.04]

        sink_flow = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])
        source_flow = Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': prices})

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='grid', imports=[source_flow]), Port(id='demand', exports=[sink_flow])],
        )

        expected = 50 * 0.02 + 50 * 0.08 + 50 * 0.04
        assert result.objective == pytest.approx(expected, abs=1e-6)


class TestContributionFrom:
    def test_contribution_from_self_reference_raises(self):
        """Self-referencing contribution_from raises ValueError."""

        source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match='cannot reference itself'):
            optimize(
                timesteps=ts(3),
                carriers=[Carrier(id='elec')],
                effects=[Effect(id='cost', contribution_from={'cost': 0.5})],
                objective='cost',
                ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
            )

    def test_contribution_from_circular_raises(self):
        """Circular contribution_from dependency raises ValueError."""

        source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04, 'co2': 0.5})
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match='Circular contribution_from dependency'):
            optimize(
                timesteps=ts(3),
                carriers=[Carrier(id='elec')],
                effects=[
                    Effect(id='cost', contribution_from={'co2': 50}),
                    Effect(id='co2', unit='kg', contribution_from={'cost': 0.01}),
                ],
                objective='cost',
                ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
            )

    def test_contribution_from_carbon_pricing(self):
        """CO2 at 0.5 kg/MWh, carbon price 50 €/kg → cost includes CO2 * 50."""
        demand = [50.0, 80.0, 60.0]

        source = Flow(
            carrier='elec',
            size=200,
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 190 MWh
        direct_cost = total_energy * 0.04
        co2_total = total_energy * 0.5
        co2_cost = co2_total * 50
        expected_cost = direct_cost + co2_cost
        assert result.objective == pytest.approx(expected_cost, abs=1e-6)

    def test_contribution_from_source_unaffected(self):
        """Source effect total is unchanged by contribution_from on target."""
        demand = [50.0, 80.0, 60.0]

        source = Flow(
            carrier='elec',
            size=200,
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        total_energy = sum(demand)
        expected_co2 = total_energy * 0.5
        co2_total = float(result.effect_totals.sel(effect='co2').values)
        assert co2_total == pytest.approx(expected_co2, abs=1e-6)

    def test_contribution_from_transitive(self):
        """PE → CO2 → cost chain: transitivity via variable chaining."""
        demand = [50.0, 80.0, 60.0]

        source = Flow(
            carrier='elec',
            size=200,
            effects_per_flow_hour={'pe': 2.0},  # 2 kWh_PE per kWh_el
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg', contribution_from={'pe': 0.3}),  # 0.3 kg_CO2/kWh_PE
                Effect(id='pe', unit='kWh'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 190 MWh
        pe_total = total_energy * 2.0  # 380
        co2_total = pe_total * 0.3  # 114
        cost_total = co2_total * 50  # 5700

        assert float(result.effect_totals.sel(effect='pe').values) == pytest.approx(pe_total, abs=1e-6)
        assert float(result.effect_totals.sel(effect='co2').values) == pytest.approx(co2_total, abs=1e-6)
        assert result.objective == pytest.approx(cost_total, abs=1e-6)

    def test_contribution_from_time_varying(self):
        """Time-varying contribution_from uses per-timestep values for temporal."""
        demand = [50.0, 80.0, 60.0]

        source = Flow(
            carrier='elec',
            size=200,
            effects_per_flow_hour={'co2': 0.5},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])

        carbon_prices = [40.0, 50.0, 60.0]
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(
                    id='cost',
                    contribution_from={'co2': carbon_prices},  # time-varying
                ),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        # per_ts[co2, t] = demand[t] * 0.5 (dt=1)
        # per_ts[cost, t] = carbon_price[t] * per_ts[co2, t]
        # total[cost] = sum(per_ts[cost, t])  (no lump costs)
        expected = sum(d * 0.5 * p for d, p in zip(demand, carbon_prices, strict=True))
        assert result.objective == pytest.approx(expected, abs=1e-6)

    def test_contribution_from_investment(self):
        """Sizing CO2 priced into cost via contribution_from."""
        demand = [50.0, 50.0, 50.0]

        source = Flow(
            carrier='elec',
            size=Sizing(size_min=50, size_max=200, mandatory=True, effects_per_size={'co2': 10}),
            effects_per_flow_hour={'cost': 0.04, 'co2': 0.5},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 150 MWh
        # Solver picks min feasible size = 50 MW (demand = 50 MW)
        invest_size = 50.0
        direct_cost = total_energy * 0.04
        op_co2 = total_energy * 0.5
        invest_co2 = invest_size * 10
        co2_total = op_co2 + invest_co2
        # cost = direct_cost + 50 * (op_co2 per_ts contribution summed) + 50 * invest_co2
        cost_total = direct_cost + op_co2 * 50 + invest_co2 * 50

        assert float(result.effect_totals.sel(effect='co2').values) == pytest.approx(co2_total, abs=1e-6)
        assert result.objective == pytest.approx(cost_total, abs=1e-6)

    def test_contribution_from_investment_transitive(self):
        """PE → CO2 → cost: 3-level chain with investment costs propagates correctly."""
        demand = [50.0, 50.0, 50.0]

        source = Flow(
            carrier='elec',
            size=Sizing(size_min=50, size_max=200, mandatory=True, effects_per_size={'pe': 5}),
            effects_per_flow_hour={'pe': 2.0},
        )
        sink = Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])

        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[
                Effect(id='cost', contribution_from={'co2': 50}),
                Effect(id='co2', unit='kg', contribution_from={'pe': 0.3}),
                Effect(id='pe', unit='kWh'),
            ],
            objective='cost',
            ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[sink])],
        )

        total_energy = sum(demand)  # 150 MWh
        invest_size = 50.0

        # Temporal chain: pe_op=300, co2_op=90, cost_op=4500
        pe_op = total_energy * 2.0  # 300
        co2_op = pe_op * 0.3  # 90
        cost_op = co2_op * 50  # 4500

        # Periodic chain: pe_inv=250, co2_inv=75, cost_inv=3750
        pe_inv = invest_size * 5  # 250
        co2_inv = pe_inv * 0.3  # 75
        cost_inv = co2_inv * 50  # 3750

        pe_total = pe_op + pe_inv  # 550
        co2_total = co2_op + co2_inv  # 165
        cost_total = cost_op + cost_inv  # 8250

        assert float(result.effect_totals.sel(effect='pe').values) == pytest.approx(pe_total, abs=1e-4)
        assert float(result.effect_totals.sel(effect='co2').values) == pytest.approx(co2_total, abs=1e-4)
        assert result.objective == pytest.approx(cost_total, abs=1e-4)


class TestPenaltyEffect:
    def test_penalty_always_in_objective(self):
        """The built-in penalty effect is minimized without being named.

        Two sources with identical cost 1; SrcB carries penalty=1. Demand=10.
        The penalty breaks the tie: all flow from SrcA. Objective includes
        the (zero) penalty: 10.
        """
        result = optimize(
            ts(1),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10])]),
                Port(id='SrcA', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})]),
                Port(id='SrcB', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'penalty': 1})]),
            ],
        )
        assert_allclose(result.objective, 10.0, rtol=1e-5)
        assert_allclose(result.flow_rate('SrcA(Heat)').values, [10.0], rtol=1e-5)
        assert_allclose(result.flow_rate('SrcB(Heat)').values, [0.0], atol=1e-6)

    def test_penalty_contributes_to_objective_value(self):
        """Incurred penalty is part of the objective value.

        Single source with cost 1 and penalty 0.5. Demand=10.
        objective = 10 * (1 + 0.5) = 15; cost total stays 10.
        """
        result = optimize(
            ts(1),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10])]),
                Port(id='Src', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'penalty': 0.5})]),
            ],
        )
        assert_allclose(result.objective, 15.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-5)

    def test_penalty_weight_zero_ignores_penalty(self):
        """Naming penalty at weight 0 solves without the penalty term.

        Same model as above: objective = 10 (cost only), penalty tracked
        but not minimized.
        """
        result = optimize(
            ts(1),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective={'cost': 1.0, 'penalty': 0.0},
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10])]),
                Port(id='Src', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'penalty': 0.5})]),
            ],
        )
        assert_allclose(result.objective, 10.0, rtol=1e-5)

    def test_penalty_weight_scales_term(self):
        """A penalty weight in the dict scales its objective contribution.

        cost 1 + penalty 0.5 per flow-hour, weight 2: objective =
        10 + 2 * 5 = 20; cost total stays 10.
        """
        result = optimize(
            ts(1),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective={'cost': 1.0, 'penalty': 2.0},
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10])]),
                Port(id='Src', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'penalty': 0.5})]),
            ],
        )
        assert_allclose(result.objective, 20.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-5)


class TestWeightedObjective:
    def test_objective_dict_weights(self):
        """Dict form weights effects in the objective without touching totals.

        Dirty (cost 1, co2 1) vs Clean (cost 20, co2 0), demand 10.
        Weighted: dirty = 1 + 50*1 = 51/MWh, clean = 20/MWh -> Clean wins.
        objective = 200; tracked cost = 200, co2 = 0.

        Sensitivity: With objective='cost', Dirty wins (objective 10).
        """
        result = optimize(
            ts(1),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='co2', unit='kg')],
            objective={'cost': 1.0, 'co2': 50.0},
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=[10])]),
                Port(id='Dirty', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'co2': 1})]),
                Port(id='Clean', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 20, 'co2': 0})]),
            ],
        )
        assert_allclose(result.objective, 200.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 200.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='co2').item(), 0.0, atol=1e-6)
        # Provenance: the resolved weights are recorded on the result
        assert result.objective_weights == {'cost': 1.0, 'co2': 50.0, 'penalty': 1.0}
