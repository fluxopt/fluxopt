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

from fluxopt.model_data import _family_arrays, _split_rows

if TYPE_CHECKING:
    from fluxopt.model_data import EffectFamily, FlowsData, ModelData, SizingData


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


def _dense_channel(
    coeffs: EffectFamily,
    sol_values: xr.DataArray | None,
    entity_dim: str,
    contributor_ids: list[str],
    effect_ids: list[str],
) -> xr.DataArray | float:
    """Densify one lump channel: rows x solved values, zero-filled.

    Each stacked array's rows are multiplied by the solved variable slice
    they select (via their entity label), then unstacked onto
    ``(contributor, effect[, ...])`` — pairs are unique per family, so the
    unstack cannot collide.

    Args:
        coeffs: Channel coefficients (dict of signatures or single array).
        sol_values: Solved variable with dim *entity_dim*, or None.
        entity_dim: Entity dim of *sol_values* and label coord on the rows.
        contributor_ids: Full contributor index for the result.
        effect_ids: All effect ids.

    Returns:
        Dense channel total, or ``0.0`` when the channel is empty.
    """
    total: xr.DataArray | float = 0.0
    if sol_values is None:
        return total
    for coeff in _family_arrays(coeffs):
        labels = [str(v) for v in coeff.coords[entity_dim].values]
        sel = xr.DataArray(labels, dims=['contribution'])
        vals = coeff.drop_vars([entity_dim, 'effect']) * sol_values.sel({entity_dim: sel}).drop_vars(
            entity_dim, errors='ignore'
        )
        vals = vals.assign_coords(
            contributor=('contribution', labels),
            effect=('contribution', [str(v) for v in coeff.coords['effect'].values]),
        )
        dense = (
            vals.set_index(contribution=['contributor', 'effect'])
            .unstack('contribution', fill_value=0.0)
            .reindex(contributor=contributor_ids, effect=effect_ids, fill_value=0.0)
        )
        total = total + dense
    return total


def _dense_constant(
    coeff: xr.DataArray,
    entity_dim: str,
    contributor_ids: list[str],
    effect_ids: list[str],
) -> xr.DataArray:
    """Densify constant rows (mandatory fixed costs): no variable multiplier."""
    vals = coeff.drop_vars([entity_dim, 'effect']).assign_coords(
        contributor=('contribution', [str(v) for v in coeff.coords[entity_dim].values]),
        effect=('contribution', [str(v) for v in coeff.coords['effect'].values]),
    )
    return (
        vals.set_index(contribution=['contributor', 'effect'])
        .unstack('contribution', fill_value=0.0)
        .reindex(contributor=contributor_ids, effect=effect_ids, fill_value=0.0)
    )


def _compute_sizing_lump(
    sizing: SizingData | None,
    solution: xr.Dataset,
    contributor_ids: list[str],
    effect_ids: list[str],
    entity_dim: str,
    size_var: str,
    indicator_var: str,
) -> xr.DataArray | float:
    """Compute Sizing lump contributions for flows or storages.

    Args:
        sizing: Sizing arrays for this entity family, or None.
        solution: Solved variable dataset.
        contributor_ids: Contributor ids for this entity type.
        effect_ids: All effect ids.
        entity_dim: Entity dim name (``flow`` or ``storage``).
        size_var: Solution variable name for size.
        indicator_var: Solution variable name for binary indicator.
    """
    if sizing is None:
        return 0.0
    result = _dense_channel(sizing.effects_per_size, solution.get(size_var), entity_dim, contributor_ids, effect_ids)
    if sizing.effects_fixed is not None:
        mandatory = sizing.mandatory
        lookup = dict(zip(mandatory.coords[mandatory.dims[0]].values, mandatory.values, strict=True))
        optional_rows, mandatory_rows = _split_rows(sizing.effects_fixed, entity_dim, lookup)
        result = result + _dense_channel(
            optional_rows, solution.get(indicator_var), entity_dim, contributor_ids, effect_ids
        )
        if mandatory_rows is not None:
            result = result + _dense_constant(mandatory_rows, entity_dim, contributor_ids, effect_ids)
    return result


def _compute_investment_lump(
    fds: FlowsData,
    solution: xr.Dataset,
    flow_ids: list[str],
    effect_ids: list[str],
) -> xr.DataArray | float:
    """Compute Investment lump contributions per (flow, effect).

    Each of the 4 Investment cost parameters multiplies a different solver
    variable, but all accumulate into the same lump bucket per flow.
    """
    if fds.invest is None:
        return 0.0
    channels = [
        (fds.invest.effects_per_size_at_build, 'invest--size_at_build'),
        (fds.invest.effects_fixed_at_build, 'invest--build'),
        (fds.invest.effects_per_size_recurring, 'flow--size'),  # rows select the invest flows
        (fds.invest.effects_fixed_recurring, 'invest--active'),
    ]
    result: xr.DataArray | float = 0.0
    for coeffs, var_name in channels:
        result = result + _dense_channel(coeffs, solution.get(var_name), 'flow', flow_ids, effect_ids)
    return result


def _add_temporal_rows(
    target: xr.DataArray,
    coeffs: EffectFamily,
    sol_values: xr.DataArray | None,
    entity_dim: str,
    scale: xr.DataArray | None = None,
    remap: dict[str, str] | None = None,
) -> None:
    """Accumulate one temporal channel into the dense per-flow view, in place.

    Each row's coefficient is multiplied by the solved variable slice it
    selects (x *scale* when given, e.g. dt) and added into its
    (contributor flow, effect) cell via unbuffered ``np.add.at`` — rows from
    different channels (or components remapped onto the same flow) may hit
    the same cell.

    Args:
        target: Dense ``(flow, effect, time[, period])`` accumulator.
        coeffs: Signature-grouped channel coefficients.
        sol_values: Solved variable with dim *entity_dim*, or None.
        entity_dim: Entity dim of *sol_values* and label coord on the rows.
        scale: Optional per-timestep multiplier (dt).
        remap: Optional entity id -> contributor flow id (component status
            attributes to the first governed flow); unmapped rows are skipped.
    """
    if sol_values is None:
        return
    extra_dims = [d for d in target.dims if d not in ('flow', 'effect')]
    flow_pos = {str(v): i for i, v in enumerate(target.coords['flow'].values)}
    eff_pos = {str(v): i for i, v in enumerate(target.coords['effect'].values)}
    for coeff in _family_arrays(coeffs):
        labels = [str(v) for v in coeff.coords[entity_dim].values]
        contributors = [remap.get(lb) if remap is not None else lb for lb in labels]
        keep = [i for i, c in enumerate(contributors) if c is not None and c in flow_pos]
        if not keep:
            continue
        sub = coeff.isel(contribution=keep)
        sel = xr.DataArray([labels[i] for i in keep], dims=['contribution'])
        vals = sub.drop_vars([entity_dim, 'effect']) * sol_values.sel({entity_dim: sel}).drop_vars(
            entity_dim, errors='ignore'
        )
        if scale is not None:
            vals = vals * scale
        vals = vals.transpose('contribution', *extra_dims)
        rows_f = [flow_pos[str(contributors[i])] for i in keep]
        rows_e = [eff_pos[str(v)] for v in sub.coords['effect'].values]
        np.add.at(target.values, (rows_f, rows_e), vals.values)


def _compute_direct(solution: xr.Dataset, data: ModelData) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Compute direct (no cross-effect propagation) per-contributor temporal and lump.

    Returns ``(temporal, lump, all_ids)`` where each contributor's effects are
    only those it directly emits — independent of ``contribution_from`` chains.
    """
    flow_ids: list[str] = list(data.flows.size.coords['flow'].values)
    effect_ids: list[str] = list(data.effects.total_min.coords['effect'].values)
    stor_ids: list[str] = list(data.storages.capacity.coords['storage'].values) if data.storages is not None else []
    all_ids = flow_ids + stor_ids

    rate = solution['flow--rate']  # (flow, time)
    dt = data.dims.dt  # (time,)

    rate_dt = rate * dt
    extra_dims = [d for d in rate_dt.dims if d != 'flow']
    temporal_flow = xr.DataArray(
        np.zeros((len(flow_ids), len(effect_ids), *(rate_dt.sizes[d] for d in extra_dims))),
        dims=['flow', 'effect', *extra_dims],
        coords={
            'flow': flow_ids,
            'effect': effect_ids,
            **{d: rate_dt.coords[d] for d in extra_dims if d in rate_dt.coords},
        },
    )

    # Component status attributes to the first governed flow
    fds = data.flows
    first_flow_per_comp: dict[str, str] = {}
    if fds.cstatus is not None and fds.cstatus.governed_flows is not None:
        gf = fds.cstatus.governed_flows
        first_flow_per_comp = {
            str(comp_id): str(gf.sel(cstatus_component=comp_id).values[0])
            for comp_id in gf.coords['cstatus_component'].values
            if str(gf.sel(cstatus_component=comp_id).values[0])
        }

    _add_temporal_rows(temporal_flow, fds.effect_coeff, rate, 'flow', scale=dt)
    if fds.status is not None:
        _add_temporal_rows(temporal_flow, fds.status.effects_running, solution.get('flow--on'), 'flow', scale=dt)
        _add_temporal_rows(temporal_flow, fds.status.effects_startup, solution.get('flow--startup'), 'flow')
    if fds.cstatus is not None:
        _add_temporal_rows(
            temporal_flow,
            fds.cstatus.effects_running,
            solution.get('component--on'),
            'component',
            scale=dt,
            remap=first_flow_per_comp,
        )
        _add_temporal_rows(
            temporal_flow,
            fds.cstatus.effects_startup,
            solution.get('component--startup'),
            'component',
            remap=first_flow_per_comp,
        )

    temporal = temporal_flow.rename({'flow': 'contributor'})

    # --- Lump: flow sizing + investment costs ---
    flow_lump = _compute_sizing_lump(
        fds.sizing,
        solution,
        flow_ids,
        effect_ids,
        'flow',
        'flow--size',
        'flow--size_indicator',
    )
    flow_lump = flow_lump + _compute_investment_lump(fds, solution, flow_ids, effect_ids)
    flow_lump = _as_dense_lump(flow_lump, flow_ids, effect_ids)

    # --- Lump: storage sizing costs ---
    if stor_ids:
        assert data.storages is not None
        stor_lump = _compute_sizing_lump(
            data.storages.sizing,
            solution,
            stor_ids,
            effect_ids,
            'storage',
            'storage--capacity',
            'storage--size_indicator',
        )
        lump = xr.concat([flow_lump, _as_dense_lump(stor_lump, stor_ids, effect_ids)], dim='contributor')
    else:
        lump = flow_lump

    return temporal, lump, all_ids


def _as_dense_lump(lump: xr.DataArray | float, contributor_ids: list[str], effect_ids: list[str]) -> xr.DataArray:
    """Guarantee a dense (contributor, effect) array even for empty channel sums."""
    if isinstance(lump, xr.DataArray):
        return lump
    return xr.DataArray(
        np.zeros((len(contributor_ids), len(effect_ids))),
        dims=['contributor', 'effect'],
        coords={'contributor': contributor_ids, 'effect': effect_ids},
    )


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
    solver = solution['effect--total']
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
