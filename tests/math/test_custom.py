from __future__ import annotations

import mathspec as ms
import polars as pl
import pytest
import specsolve
from conftest import read, ts

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port, optimize

GRID_CAP = """
parameters:
  grid_cap: {dims: [time]}
  is_grid: {dims: [flow], dtype: bool}
constraints:
  grid_cap_row:
    dims: [flow, time, period]
    where: is_grid
    expression: rate <= grid_cap
"""

GRID_ENERGY = """
expressions:
  grid_energy:
    expression: sum(rate * dt, over=flow)
    description: every flow the system moved, per step
"""


class TestEditingTheMath:
    """Extending the model by editing the spec and its sources, then solving with specsolve.

    `customize` used to hand a caller the built linopy model to poke. There is
    no such object now: the math is a spec and the numbers are tables, so a
    caller edits either and hands both to `specsolve.solve`.
    """

    @pytest.fixture
    def simple_system(self):
        """Single-bus system: grid source (size=100) feeding a fixed 50 MW demand."""
        return {
            'timesteps': ts(3),
            'carriers': [Carrier(id='elec')],
            'effects': [Effect(id='cost')],
            'objective': 'cost',
            'ports': [
                Port(id='grid', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
                Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
            ],
        }

    def test_the_spec_reads_without_data_or_solver(self, simple_system):
        """The equations are an artefact before anything is bound to them."""
        spec = FlowSystem(**simple_system).spec()

        assert 'carrier_balance' in spec.constraints
        assert spec.objective.sense == 'minimize'
        assert 'carrier_balance' in spec.to_yaml(), 'it round-trips as the file a reviewer reads'

    def test_an_added_constraint_changes_the_answer(self, simple_system):
        """A row the caller wrote, in the same language, with its own data."""
        # A dearer second source, so capping the cheap one has somewhere to go
        # rather than making a fixed demand infeasible.
        simple_system['ports'].append(
            Port(id='backup', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 5.0})])
        )
        base = optimize(**simple_system)
        assert read(base, 'rate').sel(flow='grid(elec)').values == pytest.approx([50.0] * 3, abs=1e-6)

        system = FlowSystem(**simple_system)
        spec = ms.override(system.spec(), [GRID_CAP])
        sources = system.sources() | {
            'grid_cap': pl.DataFrame({'time': ts(3), 'value': [30.0, 30.0, 30.0]}),
            'is_grid': pl.DataFrame({'flow': ['grid(elec)'], 'value': [True]}),
        }

        result = specsolve.solve(spec, sources)
        # The cap binds: 50 was the unconstrained answer, 30 is the cap, and
        # the dearer source picks up the rest.
        assert read(result, 'rate').sel(flow='grid(elec)').values == pytest.approx([30.0] * 3, abs=1e-6)
        assert read(result, 'rate').sel(flow='backup(elec)').values == pytest.approx([20.0] * 3, abs=1e-6)
        assert result.objective > base.objective

    def test_an_edited_table_changes_the_answer(self, simple_system):
        """The sources are the caller's to edit: a dearer grid costs more."""
        system = FlowSystem(**simple_system)
        sources = system.sources()
        rates = sources['effects_per_flow_hour']
        sources['effects_per_flow_hour'] = rates.with_columns(rates['value'] * 3)

        assert specsolve.solve(system.spec(), sources).objective == pytest.approx(3 * 150.0, abs=1e-6)

    def test_the_unedited_pair_answers_what_optimize_answers(self, simple_system):
        """`optimize` is `specsolve.solve(spec, sources)`, not a different model."""
        system = FlowSystem(**simple_system)
        assert specsolve.solve(system.spec(), system.sources()).objective == pytest.approx(
            optimize(**simple_system).objective, abs=1e-9
        )

    def test_a_caller_s_own_expression_comes_back_and_survives_the_archive(self, simple_system, tmp_path):
        """Naming a quantity is how you ask for it; the archive carries it home."""
        system = FlowSystem(**simple_system)
        spec = ms.override(system.spec(), [GRID_ENERGY])

        archive = tmp_path / 'run.zip'
        result = specsolve.solve(spec, system.sources(), archive=archive)
        # both sides of the bus: 50 imported and 50 exported, each hour
        assert read(result, 'grid_energy', 'expression').values == pytest.approx([100.0] * 3, abs=1e-6)

        back = specsolve.load_archive(archive).answer
        assert read(back, 'grid_energy', 'expression').values == pytest.approx([100.0] * 3, abs=1e-6)
