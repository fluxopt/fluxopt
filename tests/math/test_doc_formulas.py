"""Structural verification of the equations in ``docs/math/``.

Each test corresponds to one equation in the math docs and pins the
**coefficients** linopy emits — not just the solved objective value.
Renaming a term or forgetting a ``dt`` factor fails the test with a pointer
to the equation it verifies.

Convention: every equation verified here is annotated in the doc with
``<!-- verified-by: tests/math/test_doc_formulas.py::<test_name> -->``.
``tests/test_doc_anchors.py`` enforces that the anchor resolves.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pytest

from fluxopt import Carrier, Effect, Flow, Port
from fluxopt.model import FlowSystem
from fluxopt.model_data import ModelData


def _term_coeff(constraint: Any, row: dict[str, Any], var_label: int) -> float:
    """Return the coefficient on ``var_label`` in a single row of a linopy constraint.

    A linopy constraint stores ``coeffs`` and ``vars`` as parallel arrays indexed
    by ``_term``. This looks up which term slot holds ``var_label`` in the given
    row and returns the coefficient — or 0.0 if the variable does not appear.
    """
    vars_row = constraint.vars.sel(row).values
    coeffs_row = constraint.coeffs.sel(row).values
    idx = np.where(vars_row == var_label)[0]
    return float(coeffs_row[idx[0]]) if len(idx) else 0.0


def _build(dt_hours: float = 2.0, cost_coeff: float = 30.0, co2_to_cost: float = 50.0) -> FlowSystem:
    """Minimal 2-timestep system with a CO2 → cost cross-effect."""
    timesteps = [datetime(2024, 1, 1, h * int(dt_hours)) for h in range(2)]
    data = ModelData.build(
        timesteps=timesteps,
        carriers=[Carrier('Heat')],
        effects=[
            Effect('cost', contribution_from={'co2': co2_to_cost}),
            Effect('co2'),
        ],
        ports=[
            Port('Demand', exports=[Flow('Heat', size=1, fixed_relative_profile=[5, 5])]),
            Port('Src', imports=[Flow('Heat', effects_per_flow_hour={'cost': cost_coeff, 'co2': 0.2})]),
        ],
    )
    model = FlowSystem(data)
    model._objective_effects = ['cost']
    model.build()
    return model


class TestTemporalEquation:
    """Verifies ``docs/math/effects.md`` -- Phi^temporal_{k,t} equation.

    Phi^temporal_{k,t} = sum_f c_{f,k,t} * P_{f,t} * dt_t
                       + sum_j alpha_{k,j,t} * Phi^temporal_{j,t}

    linopy rearranges this to ``effect_temporal - (sum_f c*dt*P + sum_j alpha*Phi_j) = 0``,
    so LHS coefficients are ``+1`` on Phi_k, ``-c*dt`` on flows, ``-alpha`` on
    source-effect variables; RHS is 0.
    """

    def test_effect_temporal_equation(self) -> None:
        dt = 2.0
        cost_coeff = 30.0
        alpha = 50.0
        model = _build(dt_hours=dt, cost_coeff=cost_coeff, co2_to_cost=alpha)

        constraint = model.m.constraints['effect_temporal_eq']
        t0 = model.data.dims.time.values[0]
        row = {'effect': 'cost', 'time': t0}

        et_cost = int(model.effect_temporal.labels.sel(effect='cost', time=t0).item())
        et_co2 = int(model.effect_temporal.labels.sel(effect='co2', time=t0).item())
        src = int(model.flow_rate.labels.sel(flow='Src(Heat)', time=t0).item())
        demand = int(model.flow_rate.labels.sel(flow='Demand(Heat)', time=t0).item())

        # Phi^temporal_{cost, t0}: coefficient on itself is +1
        assert _term_coeff(constraint, row, et_cost) == pytest.approx(1.0)

        # Flow term: -c_{Src,cost} * dt
        assert _term_coeff(constraint, row, src) == pytest.approx(-cost_coeff * dt)

        # Cross-effect term: -alpha_{cost,co2}
        assert _term_coeff(constraint, row, et_co2) == pytest.approx(-alpha)

        # Demand flow has no cost coefficient -- should not contribute
        assert _term_coeff(constraint, row, demand) == pytest.approx(0.0)

        # Equality constraint with RHS = 0
        assert constraint.sign.sel(row).item() == '='
        assert float(constraint.rhs.sel(row)) == pytest.approx(0.0)
