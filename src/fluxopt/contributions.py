"""Per-contributor effect breakdown.

Decomposes solver effect totals into per-contributor (flow/storage) parts,
split into temporal (per-timestep) and lump (sizing + investment costs) domains —
matching the model's own temporal/lump structure.

Two views are supported via the ``cross_effects`` parameter on
``compute_effect_contributions``:

- **with cross-effects** (default): propagates ``contribution_from`` chains
  via the Leontief inverse — ``total = (I - C)^-1 * direct`` — so each
  contributor is charged the full priced-in cost (e.g. CO₂ → cost).
- **direct**: skips Leontief; each contributor shows only effects it
  directly emits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from fluxopt.contract import Dim, Var

if TYPE_CHECKING:
    from fluxopt.model_data import FlowsData, ModelData, SizingData


def _leontief(cf: xr.DataArray) -> xr.DataArray:
    """Compute Leontief inverse (I - C)^-1 from cross-effect coefficients.

    Args:
        cf: Cross-effect coefficients with dims ``(effect, source_effect)``
            and optionally extra batch dims (e.g. ``time``).
    """
    n = cf.sizes['effect']
    other_dims = [d for d in cf.dims if d not in ('effect', 'source_effect')]
    ordered = [*other_dims, 'effect', 'source_effect']
    vals = cf.transpose(*ordered).values  # (..., n, n)
    mat = np.eye(n) - vals
    if np.any(np.linalg.matrix_rank(mat) < n):
        raise ValueError('Cross-effect matrix (I - C) is singular — check for circular contribution_from chains')
    inv = np.linalg.inv(mat)  # (..., n, n)
    return xr.DataArray(inv, dims=ordered, coords=cf.coords).transpose(*cf.dims)


def _apply_leontief(
    leontief: xr.DataArray,
    arr: xr.DataArray,
) -> xr.DataArray:
    """Apply Leontief inverse to an array with an ``effect`` dimension.

    Args:
        leontief: Leontief inverse ``(effect, source_effect[, ...])``.
        arr: Array whose ``effect`` dim is contracted over.
    """
    result: xr.DataArray = xr.dot(leontief, arr.rename({'effect': 'source_effect'}), dim='source_effect')
    return result


def _compute_sizing_lump(
    sizing: SizingData | None,
    solution: xr.Dataset,
    contributor_ids: list[str],
    effect_ids: list[str],
    sizing_dim: str,
    entity_dim: str,
    size_var: str,
    indicator_var: str,
) -> xr.DataArray:
    """Compute Sizing lump contributions for flows or storages.

    Args:
        sizing: The entity family's ``SizingData`` container, or None.
        solution: Solved variable dataset.
        contributor_ids: Contributor ids for this entity type.
        effect_ids: All effect ids.
        sizing_dim: Sizing dimension name (``sizing_flow`` or ``sizing_storage``).
        entity_dim: Entity dimension name (``flow`` or ``storage``).
        size_var: Solution variable name for size.
        indicator_var: Solution variable name for binary indicator.
    """
    result = xr.DataArray(
        np.zeros((len(contributor_ids), len(effect_ids))),
        dims=['contributor', 'effect'],
        coords={'contributor': contributor_ids, 'effect': effect_ids},
    )
    if sizing is None:
        return result
    rename = {sizing_dim: entity_dim}

    # Per-size costs
    if size_var in solution:
        eps = sizing.effects_per_size.rename(rename)
        term = (eps * solution[size_var]).reindex({entity_dim: contributor_ids}, fill_value=0.0)
        result = result + term.rename({entity_dim: 'contributor'})

    # Fixed costs — optional (binary indicator * cost)
    if indicator_var in solution:
        indicator = solution[indicator_var].dropna(entity_dim)
        opt_ids = list(indicator.coords[entity_dim].values)
        ef = sizing.effects_fixed.rename(rename).reindex({entity_dim: opt_ids}, fill_value=0.0)
        term = (ef * indicator).reindex({entity_dim: contributor_ids}, fill_value=0.0)
        result = result + term.rename({entity_dim: 'contributor'})

    # Fixed costs — mandatory (constant)
    mand_mask = sizing.mandatory.values
    if mand_mask.any():
        mand_ids = list(sizing.mandatory.coords[sizing_dim].values[mand_mask])
        ef_mand = sizing.effects_fixed.sel({sizing_dim: mand_ids}).rename(rename)
        term = ef_mand.reindex({entity_dim: contributor_ids}, fill_value=0.0)
        result = result + term.rename({entity_dim: 'contributor'})

    return result


def _compute_investment_lump(
    fds: FlowsData,
    solution: xr.Dataset,
    flow_ids: list[str],
    effect_ids: list[str],
) -> xr.DataArray:
    """Compute Investment lump contributions per (flow, effect).

    Each of the 4 Investment cost parameters multiplies a different solver
    variable, but all accumulate into the same lump bucket per flow.
    """
    result = xr.DataArray(
        np.zeros((len(flow_ids), len(effect_ids))),
        dims=['contributor', 'effect'],
        coords={'contributor': flow_ids, 'effect': effect_ids},
    )
    if fds.invest is None:
        return result
    pairs = [
        (fds.invest.effects_per_size_at_build, Var.INVEST_SIZE_AT_BUILD),
        (fds.invest.effects_fixed_at_build, Var.INVEST_BUILD),
        (fds.invest.effects_per_size_recurring, Var.FLOW_SIZE),  # selected to invest flows
        (fds.invest.effects_fixed_recurring, Var.INVEST_ACTIVE),
    ]
    for coeff, var_name in pairs:
        c = coeff.rename({Dim.INVEST_FLOW: 'flow'})
        var = solution[var_name]
        if var_name == Var.FLOW_SIZE:
            var = var.sel(flow=list(coeff.coords[Dim.INVEST_FLOW].values))
        term = (c * var).reindex(flow=flow_ids, fill_value=0.0)
        result = result + term.rename({'flow': 'contributor'})
    return result


def _compute_direct(solution: xr.Dataset, data: ModelData) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Compute direct (no cross-effect propagation) per-contributor temporal and lump.

    Returns ``(temporal, lump, all_ids)`` where each contributor's effects are
    only those it directly emits — independent of ``contribution_from`` chains.
    """
    flow_ids: list[str] = list(data.flows.effect_coeff.coords['flow'].values)
    effect_ids: list[str] = list(data.effects.total_min.coords['effect'].values)
    stor_ids: list[str] = list(data.storages.capacity.coords['storage'].values) if data.storages is not None else []
    all_ids = flow_ids + stor_ids

    rate = solution[Var.FLOW_RATE]  # (flow, time)
    dt = data.dims.dt  # (time,)

    # --- Temporal: per-flow contributions (flow, effect, time) ---
    temporal_flow = data.flows.effect_coeff * rate * dt

    # Status running costs
    if data.flows.status is not None and Var.FLOW_ON in solution:
        er = data.flows.status.effects_running.rename({Dim.STATUS_FLOW: 'flow'})
        temporal_flow = temporal_flow + (er * solution[Var.FLOW_ON] * dt).reindex(flow=flow_ids, fill_value=0.0)

    # Status startup costs
    if data.flows.status is not None and Var.FLOW_STARTUP in solution:
        es = data.flows.status.effects_startup.rename({Dim.STATUS_FLOW: 'flow'})
        temporal_flow = temporal_flow + (es * solution[Var.FLOW_STARTUP]).reindex(flow=flow_ids, fill_value=0.0)

    # Component-level status: attribute running and startup costs to first governed flow
    cst = data.flows.cstatus
    if cst is not None and cst.governed_flows is not None:
        gf = cst.governed_flows
        first_flow_per_comp = {
            str(comp_id): str(gf.sel(cstatus_component=comp_id).values[0])
            for comp_id in gf.coords[Dim.CSTATUS_COMPONENT].values
            if str(gf.sel(cstatus_component=comp_id).values[0])
        }

        if Var.COMPONENT_ON in solution:
            cer = cst.effects_running.rename({Dim.CSTATUS_COMPONENT: 'component'})
            comp_temporal = cer * solution[Var.COMPONENT_ON] * dt  # (component, effect, time)
            for comp_id, fid in first_flow_per_comp.items():
                if fid in flow_ids:
                    add = comp_temporal.sel(component=comp_id).drop_vars('component')
                    temporal_flow.loc[{'flow': fid}] = temporal_flow.sel(flow=fid) + add

        if Var.COMPONENT_STARTUP in solution:
            ces = cst.effects_startup.rename({Dim.CSTATUS_COMPONENT: 'component'})
            comp_startup = ces * solution[Var.COMPONENT_STARTUP]  # (component, effect, time)
            for comp_id, fid in first_flow_per_comp.items():
                if fid in flow_ids:
                    add = comp_startup.sel(component=comp_id).drop_vars('component')
                    temporal_flow.loc[{'flow': fid}] = temporal_flow.sel(flow=fid) + add

    temporal = temporal_flow.rename({'flow': 'contributor'})

    # --- Lump: flow sizing costs ---
    flow_lump = _compute_sizing_lump(
        data.flows.sizing,
        solution,
        flow_ids,
        effect_ids,
        Dim.SIZING_FLOW,
        'flow',
        Var.FLOW_SIZE,
        Var.FLOW_SIZE_INDICATOR,
    )

    # --- Lump: flow investment costs (at_build + recurring) ---
    flow_lump = flow_lump + _compute_investment_lump(data.flows, solution, flow_ids, effect_ids)

    # --- Lump: storage sizing costs ---
    if stor_ids:
        assert data.storages is not None
        stor_lump = _compute_sizing_lump(
            data.storages.sizing,
            solution,
            stor_ids,
            effect_ids,
            Dim.SIZING_STORAGE,
            'storage',
            Var.STORAGE_CAPACITY,
            Var.STORAGE_SIZE_INDICATOR,
        )
        lump = xr.concat([flow_lump, stor_lump], dim='contributor')
    else:
        lump = flow_lump

    return temporal, lump, all_ids


def _apply_cross_effects(
    temporal: xr.DataArray, lump: xr.DataArray, data: ModelData
) -> tuple[xr.DataArray, xr.DataArray]:
    """Propagate effects along ``contribution_from`` chains via Leontief inverse.

    Caller must ensure ``data.effects.cf_temporal is not None``. Time-varying
    ``contribution_from`` is averaged over time for the lump domain (mirroring
    the model's own treatment in ``model.py``).
    """
    assert data.effects.cf_temporal is not None
    temporal_out = _apply_leontief(_leontief(data.effects.cf_temporal), temporal)
    cf_lump = data.effects.cf_temporal.mean('time')
    lump_out = _apply_leontief(_leontief(cf_lump), lump)
    return temporal_out, lump_out


def _validate_against_solver(total: xr.DataArray, solution: xr.Dataset) -> None:
    """Sanity check: per-contributor totals must sum to solver ``effect--total``.

    Comparison is positional — coordinate misordering or mismatch is a real
    pipeline bug that should fail loudly here rather than be silently aligned.
    """
    solver = solution[Var.EFFECT_TOTAL]
    computed = total.sum('contributor')
    if not np.allclose(computed.values, solver.values, atol=1e-6):
        diff = abs(computed - solver)
        raise ValueError(
            f'Effect contributions do not sum to solver totals. Max deviation: {float(diff.max().values):.6g}'
        )


def _finalize(
    temporal: xr.DataArray,
    lump: xr.DataArray,
    all_ids: list[str],
    data: ModelData,
) -> xr.Dataset:
    """Combine temporal + lump into the public ``(temporal, lump, total)`` Dataset."""
    total = (temporal * data.dims.weights).sum('time').reindex(contributor=all_ids, fill_value=0.0) + lump.reindex(
        contributor=all_ids, fill_value=0.0
    )
    return xr.Dataset({'temporal': temporal, 'lump': lump, 'total': total})


def _with_cross_effects(direct: xr.Dataset, data: ModelData, solution: xr.Dataset) -> xr.Dataset:
    """Apply Leontief cross-effects on top of a precomputed direct contributions Dataset.

    Validates the resulting totals against solver ``effect--total``. When the model has
    no ``contribution_from`` chains, the direct Dataset is already the with-cross
    answer — we just validate and return it.

    Args:
        direct: Output of :func:`compute_effect_contributions` with ``cross_effects=False``.
        data: Model data the ``direct`` was built from.
        solution: Solved variable dataset (used for validation).
    """
    if data.effects.cf_temporal is None:
        _validate_against_solver(direct['total'], solution)
        return direct
    temporal, lump = _apply_cross_effects(direct['temporal'], direct['lump'], data)
    all_ids = list(direct['total'].coords['contributor'].values)
    out = _finalize(temporal, lump, all_ids, data)
    _validate_against_solver(out['total'], solution)
    return out


def compute_effect_contributions(
    solution: xr.Dataset,
    data: ModelData,
    *,
    cross_effects: bool = True,
) -> xr.Dataset:
    """Compute per-contributor effect breakdown from solved values.

    Decomposes effect totals into per-contributor parts on a unified
    ``contributor`` dimension (flow IDs + storage IDs).

    Args:
        solution: Solved variable dataset from ``Result.solution``.
        data: Model data used to build the optimization.
        cross_effects: When True (default), propagates effects along
            ``contribution_from`` chains via the Leontief inverse so each
            contributor is charged the full priced-in cost (e.g. CO₂ → cost).
            When False, returns *direct* contributions only — each contributor
            shows only effects it directly emits, ignoring cross-effects.

    Returns:
        Dataset with:
        - ``temporal`` (contributor, effect, time) — per-timestep contributions
        - ``lump`` (contributor, effect) — lump contributions (flows + storages)
        - ``total`` (contributor, effect) — temporal summed over time + lump

    Raises:
        ValueError: if ``cross_effects=True`` and the contributions don't
            match solver totals (a sanity check on the breakdown).
    """
    temporal, lump, all_ids = _compute_direct(solution, data)
    direct = _finalize(temporal, lump, all_ids, data)
    if cross_effects:
        return _with_cross_effects(direct, data, solution)
    return direct
