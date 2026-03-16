"""Per-contributor effect breakdown.

Decomposes solver effect totals into per-contributor (flow/storage) parts,
split into temporal (per-timestep), periodic (sizing, recurring investment),
and once (one-time investment) domains — matching the model's own structure.

Cross-effects use the Leontief inverse: total = (I - C)^-1 * direct,
where C is the cross-effect coefficient matrix. One-time costs bypass
the Leontief pass (matching the solver's ``effect_once`` variable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from fluxopt.model_data import ModelData


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
        raise ValueError('Cross-effect matrix (I - C) is singular — check for circular cross-effect chains')
    inv = np.linalg.inv(mat)  # (..., n, n)
    return xr.DataArray(inv, dims=ordered, coords=cf.coords)


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


def _compute_periodic(
    effects_per_size: xr.DataArray | None,
    effects_fixed: xr.DataArray | None,
    mandatory: xr.DataArray | None,
    solution: xr.Dataset,
    contributor_ids: list[str],
    effect_ids: list[str],
    sizing_dim: str,
    entity_dim: str,
    size_var: str,
    indicator_var: str,
) -> xr.DataArray:
    """Compute periodic contributions for flows or storages.

    Args:
        effects_per_size: Per-unit sizing costs ``(sizing_dim, effect)`` or None.
        effects_fixed: Fixed sizing costs ``(sizing_dim, effect)`` or None.
        mandatory: Boolean mask for mandatory sizing ``(sizing_dim,)`` or None.
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
    rename = {sizing_dim: entity_dim}

    # Per-size costs
    if effects_per_size is not None and size_var in solution:
        eps = effects_per_size.rename(rename)
        term = (eps * solution[size_var]).reindex({entity_dim: contributor_ids}, fill_value=0.0)
        result = result + term.rename({entity_dim: 'contributor'})

    # Fixed costs — optional (binary indicator * cost)
    if effects_fixed is not None and indicator_var in solution:
        indicator = solution[indicator_var].dropna(entity_dim)
        opt_ids = list(indicator.coords[entity_dim].values)
        ef = effects_fixed.rename(rename).reindex({entity_dim: opt_ids}, fill_value=0.0)
        term = (ef * indicator).reindex({entity_dim: contributor_ids}, fill_value=0.0)
        result = result + term.rename({entity_dim: 'contributor'})

    # Fixed costs — mandatory (constant)
    if effects_fixed is not None and mandatory is not None:
        mand_mask = mandatory.values
        if mand_mask.any():
            mand_ids = list(mandatory.coords[sizing_dim].values[mand_mask])
            ef_mand = effects_fixed.sel({sizing_dim: mand_ids}).rename(rename)
            term = ef_mand.reindex({entity_dim: contributor_ids}, fill_value=0.0)
            result = result + term.rename({entity_dim: 'contributor'})

    return result


def _pw_converter_first_flow(data: ModelData, conv_id: str) -> str | None:
    """Find the reference flow id for a piecewise converter."""
    pw = data.piecewise
    assert pw is not None
    ref = pw.ref_flow.sel(pw_converter=conv_id)
    ref_val = str(ref.values)
    if ref_val:
        return ref_val
    # Fallback to first pair_flow if ref_flow is missing
    mask = pw.pair_converter.values == conv_id
    return str(pw.pair_flow.values[np.where(mask)[0][0]]) if mask.any() else None


def _pw_attribute_to_flows(
    cost: xr.DataArray,
    conv_ids: list[str],
    data: ModelData,
    flow_ids: list[str],
    all_ids: list[str],
) -> xr.DataArray:
    """Map per-converter costs to per-flow contributions."""
    terms = []
    for cid in conv_ids:
        first_flow = _pw_converter_first_flow(data, cid)
        if first_flow and first_flow in flow_ids:
            val = cost.sel(pw_converter=cid).expand_dims(contributor=[first_flow])
            terms.append(val)
    if not terms:
        return xr.DataArray(0)
    concat = xr.concat(terms, dim='contributor')
    return concat.reindex(contributor=all_ids, fill_value=0.0)


def _add_pw_once_contributions(
    once: xr.DataArray,
    data: ModelData,
    solution: xr.Dataset,
    flow_ids: list[str],
    all_ids: list[str],
) -> xr.DataArray:
    """Add piecewise investment one-time contributions.

    Args:
        once: Current one-time contributions array.
        data: Model data.
        solution: Solved variable dataset.
        flow_ids: All flow ids.
        all_ids: All contributor ids.
    """
    pw = data.piecewise
    if pw is None:
        return once

    # Once: per-size costs * size_at_build
    if pw.invest_effects_per_size is not None and 'pw_invest--size_at_build' in solution:
        eps = pw.invest_effects_per_size.rename({'invest_pw_converter': 'pw_converter'})
        sab = solution['pw_invest--size_at_build']
        inv_ids = list(pw.invest_effects_per_size.coords['invest_pw_converter'].values)
        once = once + _pw_attribute_to_flows(eps * sab, inv_ids, data, flow_ids, all_ids)

    # Once: fixed costs * build
    if pw.invest_effects_fixed is not None and 'pw_invest--build' in solution:
        ef = pw.invest_effects_fixed.rename({'invest_pw_converter': 'pw_converter'})
        build = solution['pw_invest--build']
        inv_ids = list(pw.invest_effects_fixed.coords['invest_pw_converter'].values)
        once = once + _pw_attribute_to_flows(ef * build, inv_ids, data, flow_ids, all_ids)

    return once


def _add_pw_periodic_contributions(
    periodic: xr.DataArray,
    data: ModelData,
    solution: xr.Dataset,
    flow_ids: list[str],
    all_ids: list[str],
) -> xr.DataArray:
    """Add piecewise investment recurring contributions.

    Args:
        periodic: Current periodic contributions array.
        data: Model data.
        solution: Solved variable dataset.
        flow_ids: All flow ids.
        all_ids: All contributor ids.
    """
    pw = data.piecewise
    if pw is None:
        return periodic

    # Periodic: per-size costs * size
    if pw.invest_effects_per_size_periodic is not None and 'pw_invest--size' in solution:
        eps_p = pw.invest_effects_per_size_periodic.rename({'invest_pw_converter': 'pw_converter'})
        pw_size = solution['pw_invest--size']
        inv_ids = list(pw.invest_effects_per_size_periodic.coords['invest_pw_converter'].values)
        periodic = periodic + _pw_attribute_to_flows(eps_p * pw_size, inv_ids, data, flow_ids, all_ids)

    # Periodic: fixed costs * active
    if pw.invest_effects_fixed_periodic is not None and 'pw_invest--active' in solution:
        ef_p = pw.invest_effects_fixed_periodic.rename({'invest_pw_converter': 'pw_converter'})
        pw_active = solution['pw_invest--active']
        inv_ids = list(pw.invest_effects_fixed_periodic.coords['invest_pw_converter'].values)
        periodic = periodic + _pw_attribute_to_flows(ef_p * pw_active, inv_ids, data, flow_ids, all_ids)

    return periodic


def compute_effect_contributions(solution: xr.Dataset, data: ModelData) -> xr.Dataset:
    """Compute per-contributor effect breakdown from solved values.

    Decomposes solver totals into per-contributor parts on a unified
    ``contributor`` dimension (flow IDs + storage IDs).

    Args:
        solution: Solved variable dataset from ``Result.solution``.
        data: Model data used to build the optimization.

    Returns:
        Dataset with:
        - ``temporal`` (contributor, effect, time) — per-timestep contributions
        - ``periodic`` (contributor, effect) — periodic contributions (flows + storages)
        - ``once`` (contributor, effect) — one-time investment contributions
        - ``total`` (contributor, effect) — temporal summed over time + periodic + once
    """
    flow_ids: list[str] = list(data.flows.effect_coeff.coords['flow'].values)
    effect_ids: list[str] = list(data.effects.min_total.coords['effect'].values)
    stor_ids: list[str] = list(data.storages.capacity.coords['storage'].values) if data.storages is not None else []
    all_ids = flow_ids + stor_ids

    rate = solution['flow--rate']  # (flow, time)
    dt = data.dims.dt  # (time,)

    # --- Temporal: per-flow contributions (flow, effect, time) ---
    temporal_flow = data.flows.effect_coeff * rate * dt

    # Status running costs
    if data.flows.status_effects_running is not None and 'flow--on' in solution:
        er = data.flows.status_effects_running.rename({'status_flow': 'flow'})
        temporal_flow = temporal_flow + (er * solution['flow--on'] * dt).reindex(flow=flow_ids, fill_value=0.0)

    # Status startup costs
    if data.flows.status_effects_startup is not None and 'flow--startup' in solution:
        es = data.flows.status_effects_startup.rename({'status_flow': 'flow'})
        temporal_flow = temporal_flow + (es * solution['flow--startup']).reindex(flow=flow_ids, fill_value=0.0)

    # Component-level status running costs — attribute to first governed flow
    if data.flows.cstatus_effects_running is not None and 'component--on' in solution:
        er = data.flows.cstatus_effects_running  # (cstatus_component, effect, time)
        on = solution['component--on']  # (component, time)
        er_renamed = er.rename({'cstatus_component': 'component'})
        comp_temporal = er_renamed * on * dt  # (component, effect, time)
        # Attribute to first governed flow of each component
        if data.flows.cstatus_governed_flows is not None:
            for comp_id in comp_temporal.coords['component'].values:
                row = data.flows.cstatus_governed_flows.sel(cstatus_component=str(comp_id))
                first_flow = str(row.values[0])
                if first_flow and first_flow in flow_ids:
                    temporal_flow.loc[first_flow] = temporal_flow.sel(flow=first_flow) + comp_temporal.sel(
                        component=comp_id
                    )

    # Component-level status startup costs — attribute to first governed flow
    if data.flows.cstatus_effects_startup is not None and 'component--startup' in solution:
        es = data.flows.cstatus_effects_startup  # (cstatus_component, effect, time)
        startup = solution['component--startup']  # (component, time)
        es_renamed = es.rename({'cstatus_component': 'component'})
        comp_startup = es_renamed * startup  # (component, effect, time)
        if data.flows.cstatus_governed_flows is not None:
            for comp_id in comp_startup.coords['component'].values:
                row = data.flows.cstatus_governed_flows.sel(cstatus_component=str(comp_id))
                first_flow = str(row.values[0])
                if first_flow and first_flow in flow_ids:
                    temporal_flow.loc[first_flow] = temporal_flow.sel(flow=first_flow) + comp_startup.sel(
                        component=comp_id
                    )

    # Cross-effects on temporal via Leontief inverse
    if data.effects.cf_temporal is not None:
        temporal_flow = _apply_leontief(_leontief(data.effects.cf_temporal), temporal_flow)

    # Rename to contributor dim
    temporal = temporal_flow.rename({'flow': 'contributor'})

    # --- Periodic: flow costs ---
    flow_periodic = _compute_periodic(
        data.flows.sizing_effects_per_size,
        data.flows.sizing_effects_fixed,
        data.flows.sizing_mandatory,
        solution,
        flow_ids,
        effect_ids,
        'sizing_flow',
        'flow',
        'flow--size',
        'flow--size_indicator',
    )

    # --- Periodic: storage costs ---
    if stor_ids:
        stor_periodic = _compute_periodic(
            data.storages.sizing_effects_per_size,  # type: ignore[union-attr]
            data.storages.sizing_effects_fixed,  # type: ignore[union-attr]
            data.storages.sizing_mandatory,  # type: ignore[union-attr]
            solution,
            stor_ids,
            effect_ids,
            'sizing_storage',
            'storage',
            'storage--capacity',
            'storage--size_indicator',
        )
        periodic = xr.concat([flow_periodic, stor_periodic], dim='contributor')
    else:
        periodic = flow_periodic

    # --- Periodic: piecewise sizing costs ---
    pw_sz_eps = data.piecewise.sizing_effects_per_size if data.piecewise is not None else None
    if pw_sz_eps is not None:
        pw = data.piecewise
        assert pw is not None and pw.max_bp_size is not None
        sz_ids = list(pw_sz_eps.coords['sizing_pw_converter'].values)
        max_bp = pw.max_bp_size.sel(pw_converter=sz_ids).rename({'pw_converter': 'sizing_pw_converter'})
        pw_cost = pw_sz_eps * max_bp  # (sizing_pw_converter, effect)
        terms = []
        for cid in sz_ids:
            ff = _pw_converter_first_flow(data, cid)
            if ff is not None and ff in flow_ids:
                val = pw_cost.sel(sizing_pw_converter=cid).expand_dims(contributor=[ff])
                terms.append(val)
        if terms:
            periodic = periodic + xr.concat(terms, dim='contributor').reindex(contributor=all_ids, fill_value=0.0)

    pw_sz_ef = data.piecewise.sizing_effects_fixed if data.piecewise is not None else None
    if pw_sz_ef is not None:
        assert data.piecewise is not None
        terms = []
        for cid in list(pw_sz_ef.coords['sizing_pw_converter'].values):
            ff = _pw_converter_first_flow(data, cid)
            if ff is not None and ff in flow_ids:
                val = pw_sz_ef.sel(sizing_pw_converter=cid).expand_dims(contributor=[ff])
                terms.append(val)
        if terms:
            periodic = periodic + xr.concat(terms, dim='contributor').reindex(contributor=all_ids, fill_value=0.0)

    # --- Periodic: flow investment recurring costs ---
    if data.flows.invest_effects_per_size_periodic is not None and 'flow--size' in solution:
        eps_p = data.flows.invest_effects_per_size_periodic.rename({'invest_flow': 'flow'})
        invest_ids = list(data.flows.invest_effects_per_size_periodic.coords['invest_flow'].values)
        fs = solution['flow--size'].sel(flow=invest_ids)
        term = (eps_p * fs).reindex(flow=flow_ids, fill_value=0.0)
        periodic = periodic + term.rename({'flow': 'contributor'}).reindex(contributor=all_ids, fill_value=0.0)
    if data.flows.invest_effects_fixed_periodic is not None and 'invest--active' in solution:
        ef_p = data.flows.invest_effects_fixed_periodic.rename({'invest_flow': 'flow'})
        active = solution['invest--active']
        term = (ef_p * active).reindex(flow=flow_ids, fill_value=0.0)
        periodic = periodic + term.rename({'flow': 'contributor'}).reindex(contributor=all_ids, fill_value=0.0)

    # --- Periodic: piecewise investment recurring costs ---
    periodic = _add_pw_periodic_contributions(periodic, data, solution, flow_ids, all_ids)

    # Cross-effects on periodic via Leontief inverse
    if data.effects.cf_periodic is not None:
        periodic = _apply_leontief(_leontief(data.effects.cf_periodic), periodic)

    # --- Once: one-time investment costs (bypass Leontief) ---
    once: Any = 0

    # Flow investment one-time costs
    if data.flows.invest_effects_per_size is not None and 'invest--size_at_build' in solution:
        eps = data.flows.invest_effects_per_size.rename({'invest_flow': 'flow'})
        sab = solution['invest--size_at_build']
        term = (eps * sab).reindex(flow=flow_ids, fill_value=0.0)
        once = once + term.rename({'flow': 'contributor'}).reindex(contributor=all_ids, fill_value=0.0)
    if data.flows.invest_effects_fixed is not None and 'invest--build' in solution:
        ef = data.flows.invest_effects_fixed.rename({'invest_flow': 'flow'})
        build = solution['invest--build']
        term = (ef * build).reindex(flow=flow_ids, fill_value=0.0)
        once = once + term.rename({'flow': 'contributor'}).reindex(contributor=all_ids, fill_value=0.0)

    # Piecewise investment one-time costs
    once = _add_pw_once_contributions(once, data, solution, flow_ids, all_ids)

    # Ensure once is a proper DataArray (may still be 0 if no investment)
    if not isinstance(once, xr.DataArray):
        once = xr.DataArray(
            np.zeros((len(all_ids), len(effect_ids))),
            dims=['contributor', 'effect'],
            coords={'contributor': all_ids, 'effect': effect_ids},
        )

    # Cross-effects on once via Leontief inverse
    if data.effects.cf_once is not None:
        once = _apply_leontief(_leontief(data.effects.cf_once), once)

    # --- Total: temporal (weighted sum over time) + periodic + once ---
    total = (
        (temporal * data.dims.weights).sum('time').reindex(contributor=all_ids, fill_value=0.0)
        + periodic.reindex(contributor=all_ids, fill_value=0.0)
        + once.reindex(contributor=all_ids, fill_value=0.0)
    )

    # --- Validate: contributions must sum to solver effect totals ---
    solver = solution['effect--total']
    computed = total.sum('contributor')
    # Use xarray subtraction for automatic dim alignment
    diff = abs(computed - solver)
    if float(diff.max().values) > 1e-6:
        raise ValueError(
            f'Effect contributions do not sum to solver totals. Max deviation: {float(diff.max().values):.6g}'
        )

    return xr.Dataset({'temporal': temporal, 'periodic': periodic, 'once': once, 'total': total})
