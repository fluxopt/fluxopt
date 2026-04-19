"""Structural verification of the equations in ``docs/math/``.

Each test pins one equation by comparing the linopy constraint to an
**expected expression built from the model's own variables** — so the test
body mirrors the LaTeX line-for-line.

Convention: every equation verified here carries a sibling comment in the
doc::

    <!-- verified-by: tests/math/test_doc_formulas.py::<nodeid> -->

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


def _coeff_dict(vars_arr: Any, coeffs_arr: Any) -> dict[int, float]:
    """Collapse parallel (vars, coeffs) arrays into a ``{label: coeff}`` dict.

    Drops zero coefficients and linopy's ``-1`` padding sentinel. Sums
    duplicate-label slots so expressions that list a variable twice
    (e.g., after subtraction) canonicalize correctly.
    """
    result: dict[int, float] = {}
    for v, c in zip(vars_arr.ravel().tolist(), coeffs_arr.ravel().tolist(), strict=True):
        label = int(v)
        if label == -1:
            continue
        result[label] = result.get(label, 0.0) + float(c)
    return {k: v for k, v in result.items() if abs(v) > 1e-12}


def _label_names(model: Any) -> dict[int, str]:
    """Build ``{label: human_name}`` map across all variables in the linopy model."""
    names: dict[int, str] = {}
    for vname, var in model.variables.items():
        labels = var.labels
        for flat_idx, label in enumerate(labels.values.ravel()):
            if label == -1:
                continue
            # Recover the coord tuple for this flat index
            multi_idx = np.unravel_index(flat_idx, labels.shape)
            coord_str = ', '.join(
                f'{d}={labels.coords[d].values[i]}' for d, i in zip(labels.dims, multi_idx, strict=True)
            )
            names[int(label)] = f'{vname}[{coord_str}]'
    return names


def _pretty(d: dict[int, float], name_map: dict[int, str]) -> str:
    return '{' + ', '.join(f'{name_map.get(k, k)}: {v:g}' for k, v in sorted(d.items(), key=lambda kv: kv[0])) + '}'


def assert_row_equation(constraint: Any, *, row: dict[str, Any], lhs: Any, rhs: Any = 0) -> None:
    """Assert ``constraint`` at ``row`` represents the equation ``lhs = rhs``.

    ``lhs`` and ``rhs`` are linopy expressions (or scalars) built from the same
    model's variables. Both sides are rearranged to ``lhs - rhs = 0`` and
    compared coefficient-by-coefficient against the stored constraint — so
    the test body can mirror the equation as written in the docs, with no
    manual sign-flipping.
    """
    expected_expr = lhs - rhs
    expected = _coeff_dict(expected_expr.vars.values, expected_expr.coeffs.values)
    actual = _coeff_dict(constraint.vars.sel(row).values, constraint.coeffs.sel(row).values)

    if actual != pytest.approx(expected):
        names = _label_names(constraint.model)
        raise AssertionError(
            f'Constraint row {row} does not match equation:\n'
            f'  expected: {_pretty(expected, names)}\n'
            f'  actual:   {_pretty(actual, names)}'
        )
    assert constraint.sign.sel(row).item() == '=', (
        f'Expected equality constraint, got sign {constraint.sign.sel(row).item()!r}'
    )
    assert float(constraint.rhs.sel(row)) == pytest.approx(0.0)


@pytest.fixture
def model() -> FlowSystem:
    """Two-timestep system: Src -> Demand, 30 EUR/MWh, 0.2 kg CO2/MWh, CO2 priced at 50 EUR/kg."""
    timesteps = [datetime(2024, 1, 1, h * 2) for h in range(2)]
    data = ModelData.build(
        timesteps=timesteps,
        carriers=[Carrier('Heat')],
        effects=[
            Effect('cost', contribution_from={'co2': 50}),
            Effect('co2'),
        ],
        ports=[
            Port('Demand', exports=[Flow('Heat', size=1, fixed_relative_profile=[5, 5])]),
            Port('Src', imports=[Flow('Heat', effects_per_flow_hour={'cost': 30, 'co2': 0.2})]),
        ],
    )
    m = FlowSystem(data)
    m._objective_effects = ['cost']
    m.build()
    return m


class TestTemporalEquation:
    """docs/math/effects.md — Temporal Domain.

    Phi^temporal_{k,t} = sum_f c_{f,k,t} * P_{f,t} * dt_t
                       + sum_j alpha_{k,j,t} * Phi^temporal_{j,t}
    """

    def test_effect_temporal_equation(self, model: FlowSystem) -> None:
        Phi = model.effect_temporal
        P = model.flow_rate
        t = model.data.dims.time.values[0]
        dt = 2  # timestep duration [h]
        c = 30  # Src cost coefficient [EUR/MWh]
        alpha = 50  # carbon price [EUR/kg]

        assert_row_equation(
            model.m.constraints['effect_temporal_eq'],
            row={'effect': 'cost', 'time': t},
            lhs=Phi.sel(effect='cost', time=t),
            rhs=c * dt * P.sel(flow='Src(Heat)', time=t) + alpha * Phi.sel(effect='co2', time=t),
        )
