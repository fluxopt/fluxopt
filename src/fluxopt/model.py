from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from linopy import Model, Variable

from fluxopt.constraints.sparse import sparse_weighted_sum
from fluxopt.constraints.status import add_duration_tracking, add_switch_transitions
from fluxopt.constraints.storage import add_accumulation_constraints
from fluxopt.results import Result
from fluxopt.types import as_dataarray

if TYPE_CHECKING:
    from collections.abc import Callable

    from fluxopt.model_data import ModelData


class FlowSystem:
    # Sizing variables — None when no sizing is configured
    flow_size: Variable | None = None
    flow_size_indicator: Variable | None = None
    storage_capacity: Variable | None = None
    storage_capacity_indicator: Variable | None = None

    # Investment variables — None when no Investment is configured
    invest_size: Variable | None = None
    invest_build: Variable | None = None
    invest_active: Variable | None = None
    invest_size_at_build: Variable | None = None

    # Storage variables — None when no storages
    storage_level: Variable | None = None
    prior_storage_level: Variable | None = None

    # Effect / objective — set via optimize() or defaults to ['cost']

    # Status variables — None when no status is configured
    flow_on: Variable | None = None
    flow_startup: Variable | None = None
    flow_shutdown: Variable | None = None

    # Component-level status variables — None when no component status is configured
    component_on: Variable | None = None
    component_startup: Variable | None = None
    component_shutdown: Variable | None = None

    def __init__(self, data: ModelData) -> None:
        """Initialize the flow system optimization model.

        Args:
            data: Pre-built model data.
        """
        self.data = data
        self.m = Model()
        self._objective_effects: list[str] = []
        self._piecewise: dict[str, Any] = {}  # conv_id -> linopy.PiecewiseFormulation

    def _add_variables(
        self,
        *,
        lower: Any = None,
        upper: Any = None,
        coords: dict[str, xr.DataArray],
        name: str,
        binary: bool = False,
    ) -> Variable:
        """Add a variable with bounds auto-aligned to coords via as_dataarray.

        Args:
            lower: Lower bound (scalar, array, or DataArray).
            upper: Upper bound (scalar, array, or DataArray).
            coords: Coordinate dict mapping dim names to DataArrays.
            name: Variable name.
            binary: Create a binary variable instead.
        """
        coord_dict = {k: v.values for k, v in coords.items()}
        kwargs: dict[str, Any] = {'coords': coords, 'name': name, 'binary': binary}
        if not binary:
            if lower is not None:
                kwargs['lower'] = as_dataarray(lower, coord_dict)
            if upper is not None:
                kwargs['upper'] = as_dataarray(upper, coord_dict)
        return self.m.add_variables(**kwargs)

    def build(self) -> None:
        """Build all variables, constraints, and the objective."""
        # Phase 1: Decision variables
        self._create_flow_variables()
        self._create_sizing_variables()
        self._create_investment_variables()
        self._create_status_variables()
        self._create_component_status_variables()
        # Phase 2: Flow rate constraints
        self._constrain_flow_rates_plain()
        self._constrain_flow_rates_sizing()
        self._constrain_flow_rates_status()
        self._constrain_flow_rates_component_status()
        self._constrain_flow_aggregates()
        self._constrain_flow_ramps()
        # Phase 3: Feature constraints
        self._constrain_sizing()
        self._constrain_investment()
        self._constrain_status()
        self._constrain_component_status()
        # Phase 4: System
        self._create_balance()
        self._create_converter_constraints()
        self._create_piecewise_constraints()
        self._create_storage()
        self._create_effects()
        self._set_objective()
        self._builtin_var_names: frozenset[str] = frozenset(self.m.variables)

    def optimize(
        self,
        objective_effects: str | list[str],
        customize: Callable[[FlowSystem], None] | None = None,
        *,
        solver: str = 'highs',
        **kwargs: Any,
    ) -> Result:
        """Build, optionally customize, and solve the model.

        Args:
            objective_effects: Effect name(s) to minimize. Sum of named effect totals.
            customize: Optional callback to modify the linopy model between build and solve.
                Receives ``self``; use ``model.m`` to add variables/constraints.
            solver: Solver backend name.
            **kwargs: Passed through to ``linopy.Model.solve()``.
        """
        self._objective_effects = [objective_effects] if isinstance(objective_effects, str) else objective_effects
        self.build()
        if customize is not None:
            customize(self)
        return self.solve(solver_name=solver, **kwargs)

    def solve(self, **kwargs: Any) -> Result:
        """Solve the built model and return results.

        Thin wrapper around ``linopy.Model.solve()``. Call :meth:`build` first.

        Args:
            **kwargs: Passed through to ``linopy.Model.solve()``.
        """
        self.m.solve(**kwargs)
        return Result.from_model(self)

    def _create_flow_variables(self) -> None:
        """Create flow rate decision variables P_{f,t} >= 0."""
        ds = self.data.flows
        self.flow_rate = self.m.add_variables(
            lower=0,
            coords={'flow': ds.rel_lb.coords['flow'], **self.data.dims.coords(time=True, period=True)},
            name='flow--rate',
        )

    def _create_sizing_variables(self) -> None:
        """Create sizing decision variables for flows and storages.

        Both Sizing and Investment flows get entries in ``flow_size``.
        Investment adds extra variables (invest_size, invest_build, etc.)
        with constraints linking flow_size to invest_size * active.
        """
        fds = self.data.flows
        pc = self.data.dims.coords(period=True)

        # Collect all flow ids and upper bounds that need a flow_size variable
        all_ids: list[str] = []
        upper_parts: list[xr.DataArray] = []

        # --- Sizing flows ---
        if fds.sizing_min is not None:
            assert fds.sizing_max is not None
            assert fds.sizing_mandatory is not None
            sizing_ids = list(fds.sizing_min.coords['sizing_flow'].values)
            all_ids.extend(sizing_ids)
            upper_parts.append(fds.sizing_max.rename({'sizing_flow': 'flow'}))

            mandatory = fds.sizing_mandatory
            optional_ids = np.array(sizing_ids)[~mandatory.values]
            if len(optional_ids):
                self.flow_size_indicator = self._add_variables(
                    binary=True,
                    coords={'flow': xr.DataArray(optional_ids, dims=['flow']), **pc},
                    name='flow--size_indicator',
                )

        # --- Investment flows ---
        if fds.invest_min is not None:
            assert fds.invest_max is not None
            invest_ids = list(fds.invest_min.coords['invest_flow'].values)
            all_ids.extend(invest_ids)
            upper_parts.append(fds.invest_max.rename({'invest_flow': 'flow'}))

        # Create unified flow_size variable
        if all_ids:
            upper = xr.concat(upper_parts, dim='flow') if len(upper_parts) > 1 else upper_parts[0]
            flow_coord = xr.DataArray(all_ids, dims=['flow'])
            self.flow_size = self._add_variables(
                lower=0, upper=upper, coords={'flow': flow_coord, **pc}, name='flow--size'
            )

        # --- Storage capacity sizing ---
        sds = self.data.storages
        if sds is not None and sds.sizing_min is not None:
            assert sds.sizing_max is not None
            assert sds.sizing_mandatory is not None
            sizing_ids = sds.sizing_min.coords['sizing_storage'].values
            stor_coord = xr.DataArray(sizing_ids, dims=['storage'])
            upper = sds.sizing_max.rename({'sizing_storage': 'storage'})
            pc = self.data.dims.coords(period=True)
            self.storage_capacity = self._add_variables(
                lower=0, upper=upper, coords={'storage': stor_coord, **pc}, name='storage--capacity'
            )
            mandatory = sds.sizing_mandatory
            optional_ids = sizing_ids[~mandatory.values]
            if len(optional_ids):
                self.storage_capacity_indicator = self._add_variables(
                    binary=True,
                    coords={'storage': xr.DataArray(optional_ids, dims=['storage']), **pc},
                    name='storage--size_indicator',
                )

    def _create_investment_variables(self) -> None:
        """Create investment-specific decision variables.

        Investment flows already have entries in ``flow_size`` (created by
        ``_create_sizing_variables``).  This method adds the extra variables:
        - invest_size[flow]: chosen capacity (period-independent)
        - invest_build[flow, period]: binary, build in this period?
        - invest_active[flow, period]: binary, operational in this period?
        - invest_size_at_build[flow, period]: invest_size * build (big-M linked)
        """
        fds = self.data.flows
        if fds.invest_min is None:
            return
        assert fds.invest_max is not None

        invest_ids = list(fds.invest_min.coords['invest_flow'].values)
        pc = self.data.dims.coords(period=True)

        if not pc:
            msg = 'Investment requires multi-period optimization (periods must be specified)'
            raise ValueError(msg)

        flow_coord = xr.DataArray(invest_ids, dims=['flow'])
        upper = fds.invest_max.rename({'invest_flow': 'flow'})

        # invest_size[flow]: single capacity decision (no period dim)
        self.invest_size = self._add_variables(lower=0, upper=upper, coords={'flow': flow_coord}, name='invest--size')

        # invest_build[flow, period]: binary build indicator
        self.invest_build = self._add_variables(binary=True, coords={'flow': flow_coord, **pc}, name='invest--build')

        # invest_active[flow, period]: binary active indicator
        self.invest_active = self._add_variables(binary=True, coords={'flow': flow_coord, **pc}, name='invest--active')

        # invest_size_at_build[flow, period]: invest_size * build (big-M)
        self.invest_size_at_build = self._add_variables(
            lower=0, upper=upper, coords={'flow': flow_coord, **pc}, name='invest--size_at_build'
        )

    def _create_status_variables(self) -> None:
        """Create binary on/off variables for flows with Status."""
        ds = self.data.flows
        if ds.status_uptime_min is None:
            return

        status_ids = ds.status_uptime_min.coords['status_flow'].values
        flow_coord = xr.DataArray(status_ids, dims=['flow'])
        tp = {'flow': flow_coord, **self.data.dims.coords(time=True, period=True)}

        self.flow_on = self._add_variables(binary=True, coords=tp, name='flow--on')
        self.flow_startup = self._add_variables(binary=True, coords=tp, name='flow--startup')
        self.flow_shutdown = self._add_variables(binary=True, coords=tp, name='flow--shutdown')

    def _status_flow_ids(self) -> set[str]:
        """Return ids of flows with Status, or empty set."""
        ds = self.data.flows
        if ds.status_uptime_min is None:
            return set()
        return set(ds.status_uptime_min.coords['status_flow'].values)

    def _create_component_status_variables(self) -> None:
        """Create binary on/off variables for components with Status."""
        ds = self.data.flows
        if ds.cstatus_uptime_min is None:
            return

        comp_ids = ds.cstatus_uptime_min.coords['cstatus_component'].values
        comp_coord = xr.DataArray(comp_ids, dims=['component'])
        tp = {'component': comp_coord, **self.data.dims.coords(time=True, period=True)}

        self.component_on = self._add_variables(binary=True, coords=tp, name='component--on')
        self.component_startup = self._add_variables(binary=True, coords=tp, name='component--startup')
        self.component_shutdown = self._add_variables(binary=True, coords=tp, name='component--shutdown')

    def _governed_flows_map(self) -> dict[str, list[str]]:
        """Return ``{component_id: [flow_ids governed]}`` from data, or empty."""
        gf = self.data.flows.cstatus_governed_flows
        if gf is None:
            return {}
        result: dict[str, list[str]] = {}
        for comp_id in gf.coords['cstatus_component'].values:
            row = gf.sel(cstatus_component=comp_id).values
            result[str(comp_id)] = [str(f) for f in row if str(f)]
        return result

    def _component_status_flow_ids(self) -> set[str]:
        """Return ids of flows governed by component-level Status, or empty set."""
        return {fid for fids in self._governed_flows_map().values() for fid in fids}

    def _sizing_flow_ids(self) -> set[str]:
        """Return ids of sizing flows, or empty set."""
        if self.flow_size is None:
            return set()
        return set(self.flow_size.coords['flow'].values)

    def _constrain_flow_rates_plain(self) -> None:
        """Apply flow rate bounds for fixed-size flows without Status.

        P in [size * rel_lb, size * rel_ub] or P = size * profile.
        """
        ds = self.data.flows
        sizing_ids = self._sizing_flow_ids()
        status_ids = self._status_flow_ids()
        cstatus_ids = self._component_status_flow_ids()
        exclude = sizing_ids | status_ids | cstatus_ids

        size = ds.size
        rel_lb = ds.rel_lb
        rel_ub = ds.rel_ub
        fixed = ds.fixed_profile
        is_bounded = ds.bound_type == 'bounded'
        is_profile = ds.bound_type == 'profile'

        # Mask: has fixed size AND not sizing AND not status
        is_plain = size.notnull()
        if exclude:
            exclude_mask = xr.DataArray(
                [fid in exclude for fid in size.coords['flow'].values],
                dims=['flow'],
                coords={'flow': size.coords['flow']},
            )
            is_plain = is_plain & ~exclude_mask

        plain_bounded = is_bounded & is_plain
        if plain_bounded.any():
            mask = plain_bounded.broadcast_like(rel_lb)
            self.m.add_constraints(self.flow_rate >= size * rel_lb, name='flow_lb', mask=mask)
            self.m.add_constraints(self.flow_rate <= size * rel_ub, name='flow_ub', mask=mask)

        plain_profile = is_profile & is_plain
        if plain_profile.any():
            mask = plain_profile.broadcast_like(fixed) & fixed.notnull()
            self.m.add_constraints(self.flow_rate == size * fixed, name='flow_fix', mask=mask)

    def _constrain_flow_rates_sizing(self) -> None:
        """Apply flow rate bounds for investable flows without Status.

        P in [S * rel_lb, S * rel_ub] or P = S * profile.
        """
        if self.flow_size is None:
            return

        ds = self.data.flows
        status_ids = self._status_flow_ids()

        all_invest_ids = list(self.flow_size.coords['flow'].values)
        invest_ids = [fid for fid in all_invest_ids if fid not in status_ids]
        if not invest_ids:
            return

        fr = self.flow_rate.sel(flow=invest_ids)
        rl = ds.rel_lb.sel(flow=invest_ids)
        ru = ds.rel_ub.sel(flow=invest_ids)
        fp = ds.fixed_profile.sel(flow=invest_ids)
        inv_bounded = (ds.bound_type == 'bounded').sel(flow=invest_ids)
        inv_profile = (ds.bound_type == 'profile').sel(flow=invest_ids)
        fs = self.flow_size.sel(flow=invest_ids)

        var_mask = inv_bounded.broadcast_like(rl)
        if var_mask.any():
            self.m.add_constraints(fr >= rl * fs, name='flow_lb_sizing', mask=var_mask)
            self.m.add_constraints(fr <= ru * fs, name='flow_ub_sizing', mask=var_mask)

        if inv_profile.any():
            fix_mask = inv_profile.broadcast_like(fp) & fp.notnull()
            self.m.add_constraints(fr == fp * fs, name='flow_fix_sizing', mask=fix_mask)

    def _constrain_flow_rates_status(self) -> None:
        """Apply semi-continuous flow rate bounds for flows with Status.

        Fixed-size: P <= size * rel_ub * on, P >= size * rel_lb * on.
        Sizing: big-M formulation via _constrain_flow_rates_status_sizing().
        """
        if self.flow_on is None:
            return

        ds = self.data.flows
        all_status_ids = list(self.flow_on.coords['flow'].values)
        sizing_ids = self._sizing_flow_ids()
        overlap = sizing_ids & set(all_status_ids)

        # Dispatch status+sizing flows to dedicated method
        if overlap:
            self._constrain_flow_rates_status_sizing(sorted(overlap))

        # Status-only flows (fixed size)
        status_ids = [fid for fid in all_status_ids if fid not in overlap]
        if not status_ids:
            return

        fr = self.flow_rate.sel(flow=status_ids)
        size = ds.size.sel(flow=status_ids)
        rl = ds.rel_lb.sel(flow=status_ids)
        ru = ds.rel_ub.sel(flow=status_ids)
        fp = ds.fixed_profile.sel(flow=status_ids)
        is_bounded = (ds.bound_type == 'bounded').sel(flow=status_ids)
        is_profile = (ds.bound_type == 'profile').sel(flow=status_ids)

        stat_bounded = is_bounded & size.notnull()
        if stat_bounded.any():
            mask = stat_bounded.broadcast_like(rl)
            self.m.add_constraints(fr >= size * rl * self.flow_on, name='flow_lb_status', mask=mask)
            self.m.add_constraints(fr <= size * ru * self.flow_on, name='flow_ub_status', mask=mask)

        stat_profile = is_profile & size.notnull()
        if stat_profile.any():
            mask = stat_profile.broadcast_like(fp) & fp.notnull()
            self.m.add_constraints(fr == size * fp * self.flow_on, name='flow_fix_status', mask=mask)

    def _constrain_flow_rates_status_sizing(self, flow_ids: list[str]) -> None:
        """Apply big-M flow rate bounds for flows with both Status and Sizing.

        Three constraints decouple the binary on/off from the continuous size:
          P <= on * M⁺          (big-M: forces on=1 when P>0)
          P <= S  * rel_ub      (rate limited by invested size)
          P >= (on - 1) * M⁻ + S * rel_lb  (enforces minimum when on=1)

        Where M⁺ = size_max * rel_ub, M⁻ = size_max * rel_lb.

        Args:
            flow_ids: Flows that have both Status and Sizing.
        """
        ds = self.data.flows
        assert self.flow_on is not None
        assert self.flow_size is not None

        fr = self.flow_rate.sel(flow=flow_ids)
        on = self.flow_on.sel(flow=flow_ids)
        fs = self.flow_size.sel(flow=flow_ids)
        rl = ds.rel_lb.sel(flow=flow_ids)
        ru = ds.rel_ub.sel(flow=flow_ids)

        # size_max: combine from sizing_max and invest_max sources
        sizing_ids = set(ds.sizing_max.coords['sizing_flow'].values) if ds.sizing_max is not None else set()
        invest_ids = set(ds.invest_max.coords['invest_flow'].values) if ds.invest_max is not None else set()
        parts: list[xr.DataArray] = []
        for fid in flow_ids:
            if fid in sizing_ids:
                assert ds.sizing_max is not None
                parts.append(ds.sizing_max.sel(sizing_flow=fid).rename('flow'))
            elif fid in invest_ids:
                assert ds.invest_max is not None
                parts.append(ds.invest_max.sel(invest_flow=fid).rename('flow'))
            else:
                raise ValueError(f'Flow {fid!r} has Status+Sizing but no size_max in sizing or investment data')
        size_max = xr.DataArray([float(p) for p in parts], dims=['flow'], coords={'flow': flow_ids})

        big_m_ub = size_max * ru  # M⁺
        big_m_lb = size_max * rl  # M⁻

        self.m.add_constraints(fr <= on * big_m_ub, name='flow_ub_status_sizing_bigm')
        self.m.add_constraints(fr <= fs * ru, name='flow_ub_status_sizing_size')
        self.m.add_constraints(fr >= (on - 1) * big_m_lb + fs * rl, name='flow_lb_status_sizing')

        # on <= S: prevents on=1 when size=0, which would incorrectly
        # charge running/startup costs for a non-existent unit.
        self.m.add_constraints(on <= fs, name='flow_on_requires_size')

    def _constrain_flow_rates_component_status(self) -> None:
        """Apply gating constraints for flows governed by component-level Status.

        For each component with Status and each governed flow:
          bounded:  size * rel_lb * on <= P <= size * rel_ub * on
          profile:  P == size * profile * on

        Sizing/Investment governed flows are not yet supported and raise.
        """
        if self.component_on is None:
            return

        ds = self.data.flows
        sizing_ids = self._sizing_flow_ids()
        # Piecewise converters: their flows are already gated via the
        # add_piecewise_formulation `active=` parameter (when active=0, all
        # auxiliary weights -> 0 -> all curve flows pinned to 0). No extra
        # per-flow gating needed.
        piecewise_comps = set(self.data.piecewise.converter_ids()) if self.data.piecewise is not None else set()

        for comp_id, flow_ids in self._governed_flows_map().items():
            if comp_id in piecewise_comps:
                continue
            on = self.component_on.sel(component=comp_id)  # (time, period?)
            for fid in flow_ids:
                if fid in sizing_ids:
                    msg = (
                        f'Component {comp_id!r}: governed flow {fid!r} has Sizing/Investment, '
                        f'which is not yet supported with component-level status'
                    )
                    raise NotImplementedError(msg)

                fr = self.flow_rate.sel(flow=fid)
                bt = str(ds.bound_type.sel(flow=fid).values)
                size_val = ds.size.sel(flow=fid).values
                if np.isnan(size_val):  # pragma: no cover
                    # Defensive invariant check — callers (Storage.__post_init__, future
                    # PiecewiseConversion) ensure governed flows are sized. Reaching this branch
                    # means an invariant was violated upstream.
                    msg = f'Component {comp_id!r}: governed flow {fid!r} is unsized'
                    raise ValueError(msg)
                size = float(size_val)
                rl = ds.rel_lb.sel(flow=fid)
                ru = ds.rel_ub.sel(flow=fid)
                if bt == 'bounded':
                    self.m.add_constraints(fr >= size * rl * on, name=f'flow_lb_cstatus_{fid}')
                    self.m.add_constraints(fr <= size * ru * on, name=f'flow_ub_cstatus_{fid}')
                elif bt == 'profile':
                    fp = ds.fixed_profile.sel(flow=fid)
                    self.m.add_constraints(fr == size * fp * on, name=f'flow_fix_cstatus_{fid}')
                else:  # pragma: no cover
                    # Defensive — only 'bounded' / 'profile' / 'unsized' exist today,
                    # and 'unsized' is caught above. Reaching this branch means a new
                    # bound_type was introduced without updating this dispatch.
                    msg = f'Component {comp_id!r}: governed flow {fid!r} has unsupported bound_type {bt!r}'
                    raise ValueError(msg)

    def _constrain_flow_aggregates(self) -> None:
        """Bound per-period flow-hour aggregates.

        Flow hours (absolute):   H̲_f <= Σ_t P_{f,t}·Δt_t <= H̄_f
        Load factor (relative):  λ̲_f·S_f·T <= Σ_t P_{f,t}·Δt_t <= λ̄_f·S_f·T

        Each period is bounded independently. For Sizing/Investment flows
        the load factor multiplies the size *variable*; T = Σ_t Δt_t.

        See: docs/math/flows.md
        """
        ds = self.data.flows
        bounds = (ds.flow_hours_min, ds.flow_hours_max, ds.load_factor_min, ds.load_factor_max)
        if all(b is None for b in bounds):
            return
        w = self.data.dims.weights
        flow_hours = (self.flow_rate * w).sum('time')  # (flow[, period])

        if ds.flow_hours_min is not None:
            self.m.add_constraints(
                flow_hours >= ds.flow_hours_min, name='flow_hours_min', mask=ds.flow_hours_min.notnull()
            )
        if ds.flow_hours_max is not None:
            self.m.add_constraints(
                flow_hours <= ds.flow_hours_max, name='flow_hours_max', mask=ds.flow_hours_max.notnull()
            )

        total_duration = float(w.sum('time'))  # T [h]
        sized_flow_ids = self._sizing_flow_ids()
        for lf, name, sign in (
            (ds.load_factor_min, 'load_factor_min', 1),
            (ds.load_factor_max, 'load_factor_max', -1),
        ):
            if lf is None:
                continue
            lf_ids = list(lf.coords['flow'].values[lf.notnull().values])
            fixed_ids = [fid for fid in lf_ids if fid not in sized_flow_ids]
            var_ids = [fid for fid in lf_ids if fid in sized_flow_ids]
            if fixed_ids:
                # Fixed size: constant RHS λ·S·T
                rhs = lf.sel(flow=fixed_ids) * ds.size.sel(flow=fixed_ids) * total_duration
                lhs = flow_hours.sel(flow=fixed_ids)
                if sign > 0:
                    self.m.add_constraints(lhs >= rhs, name=name)
                else:
                    self.m.add_constraints(lhs <= rhs, name=name)
            if var_ids:
                # Sizing/Investment: size is a variable — move to LHS
                assert self.flow_size is not None
                coeff = lf.sel(flow=var_ids) * total_duration
                expr = flow_hours.sel(flow=var_ids) - coeff * self.flow_size.sel(flow=var_ids)
                if sign > 0:
                    self.m.add_constraints(expr >= 0, name=f'{name}_sized')
                else:
                    self.m.add_constraints(expr <= 0, name=f'{name}_sized')

    def _constrain_flow_ramps(self) -> None:
        """Limit flow rate changes between consecutive timesteps.

        Ramp up:   P_{f,t} - P_{f,t-1} <= r⁺_{f,t}·S_f·Δt_t + M_f·startup_{f,t}
        Ramp down: P_{f,t-1} - P_{f,t} <= r⁻_{f,t}·S_f·Δt_t + M_f·shutdown_{f,t}

        Applies from the second timestep onward; periods are independent.
        For Sizing/Investment flows the size variable enters the constraint
        (r·Δt moves to the LHS as a coefficient).

        For status flows the ramp does not bind across on/off transitions:
        the startup/shutdown binary relaxes it with M = static size bound
        (transitions are pinned to actual state changes by the
        ``status|exclusive`` constraint). Component-status flows relax via
        their component's transition binaries.

        See: docs/math/flows.md
        """
        ds = self.data.flows
        if ds.ramp_up is None and ds.ramp_down is None:
            return
        time_vals = self.flow_rate.coords['time'].values
        if len(time_vals) < 2:
            return
        coords_from_1 = time_vals[1:]
        curr = self.flow_rate.isel(time=slice(1, None))
        prev = self.flow_rate.isel(time=slice(None, -1)).assign_coords(time=coords_from_1)
        dt_t = self.data.dims.dt.isel(time=slice(1, None))
        status_ids = self._status_flow_ids()
        flow_to_comp = {fid: cid for cid, fids in self._governed_flows_map().items() for fid in fids}

        for ramp, name, delta, trans_flow, trans_comp in (
            (ds.ramp_up, 'ramp_up', curr - prev, self.flow_startup, self.component_startup),
            (ds.ramp_down, 'ramp_down', prev - curr, self.flow_shutdown, self.component_shutdown),
        ):
            if ramp is None:
                continue
            non_flow_dims = [d for d in ramp.dims if d != 'flow']
            has_ramp = ramp.notnull().any(non_flow_dims)
            ramp_ids = [str(f) for f in ramp.coords['flow'].values[has_ramp.values]]
            limit = ramp.isel(time=slice(1, None)) * dt_t  # r·Δt  (flow, time[, period])

            plain_ids = [fid for fid in ramp_ids if fid not in status_ids and fid not in flow_to_comp]
            flow_status_ids = [fid for fid in ramp_ids if fid in status_ids]
            self._add_ramp_constraints(plain_ids, name, '', limit, delta, None)
            if flow_status_ids:
                assert trans_flow is not None
                relax = trans_flow.sel(flow=flow_status_ids).isel(time=slice(1, None))
                self._add_ramp_constraints(flow_status_ids, name, '_status', limit, delta, relax)
            # Component-status flows share their component's transition binary
            comp_groups: dict[str, list[str]] = {}
            for fid in ramp_ids:
                if fid in flow_to_comp:
                    comp_groups.setdefault(flow_to_comp[fid], []).append(fid)
            for cid, fids in comp_groups.items():
                assert trans_comp is not None
                relax = trans_comp.sel(component=cid).isel(time=slice(1, None))
                self._add_ramp_constraints(fids, name, f'_cstatus_{cid}', limit, delta, relax)

    def _add_ramp_constraints(
        self,
        ids: list[str],
        name: str,
        suffix: str,
        limit: xr.DataArray,
        delta: Any,
        relax: Any,
    ) -> None:
        """Add ramp constraints for *ids*, optionally relaxed at transitions.

        Args:
            ids: Flow ids to constrain.
            name: Base constraint name (``ramp_up`` / ``ramp_down``).
            suffix: Name suffix distinguishing the status regime.
            limit: r·Δt coefficient, (flow, time[, period]).
            delta: Rate change expression over t >= 1.
            relax: Transition binary relaxing the ramp (M·relax added to the
                allowance), or None for the strict constraint.
        """
        ds = self.data.flows
        sized_flow_ids = self._sizing_flow_ids()
        fixed_ids = [fid for fid in ids if fid not in sized_flow_ids]
        var_ids = [fid for fid in ids if fid in sized_flow_ids]
        if fixed_ids:
            # Fixed size: constant RHS r·S̄·Δt
            rhs = limit.sel(flow=fixed_ids) * ds.size.sel(flow=fixed_ids)
            lhs = delta.sel(flow=fixed_ids)
            if relax is not None:
                lhs = lhs - self._flow_size_bounds(fixed_ids) * relax
            self.m.add_constraints(lhs <= rhs, name=f'flow_{name}{suffix}', mask=rhs.notnull())
        if var_ids:
            # Sizing/Investment: size is a variable — move to LHS
            assert self.flow_size is not None
            coeff = limit.sel(flow=var_ids)
            expr = delta.sel(flow=var_ids) - coeff * self.flow_size.sel(flow=var_ids)
            if relax is not None:
                expr = expr - self._flow_size_bounds(var_ids) * relax
            self.m.add_constraints(expr <= 0, name=f'flow_{name}_sized{suffix}', mask=coeff.notnull())

    def _flow_size_bounds(self, flow_ids: list[str]) -> xr.DataArray:
        """Static per-flow upper size bounds: fixed value or sizing/invest max.

        Args:
            flow_ids: Qualified flow ids; each must be sized.
        """
        fds = self.data.flows
        vals: list[float] = []
        for fid in flow_ids:
            v = fds.size.sel(flow=fid).values
            if not np.isnan(v):
                vals.append(float(v))
            elif fds.sizing_max is not None and fid in fds.sizing_max.coords['sizing_flow'].values:
                vals.append(float(fds.sizing_max.sel(sizing_flow=fid).values))
            elif fds.invest_max is not None and fid in fds.invest_max.coords['invest_flow'].values:
                vals.append(float(fds.invest_max.sel(invest_flow=fid).values))
            else:  # pragma: no cover — element validation guards sized flows
                raise ValueError(f'Flow {fid!r} has no static size bound for big-M')
        return xr.DataArray(vals, dims=['flow'], coords={'flow': flow_ids})

    def _constrain_sizing(self) -> None:
        """Constrain sizing variables: S in [min, max] gated by indicator."""
        # --- Flow sizing (Sizing only, not Investment) ---
        fds = self.data.flows
        if fds.sizing_min is not None:
            assert fds.sizing_max is not None
            assert fds.sizing_mandatory is not None
            assert self.flow_size is not None
            sizing_ids = fds.sizing_min.coords['sizing_flow'].values
            smin = fds.sizing_min.rename({'sizing_flow': 'flow'})
            mandatory = fds.sizing_mandatory

            mand_ids = sizing_ids[mandatory.values]
            if len(mand_ids):
                self.m.add_constraints(
                    self.flow_size.sel(flow=mand_ids) >= smin.sel(flow=mand_ids),
                    name='invest_mand_lb',
                )

            opt_ids = sizing_ids[~mandatory.values]
            if len(opt_ids):
                assert self.flow_size_indicator is not None
                smax = fds.sizing_max.rename({'sizing_flow': 'flow'})
                fs = self.flow_size.sel(flow=opt_ids)
                self.m.add_constraints(fs >= smin.sel(flow=opt_ids) * self.flow_size_indicator, name='invest_lb')
                self.m.add_constraints(fs <= smax.sel(flow=opt_ids) * self.flow_size_indicator, name='invest_ub')

        # --- Storage capacity sizing ---
        if self.storage_capacity is not None:
            sds = self.data.storages
            assert sds is not None
            assert sds.sizing_min is not None
            assert sds.sizing_max is not None
            assert sds.sizing_mandatory is not None
            smin = sds.sizing_min.rename({'sizing_storage': 'storage'})
            mandatory = sds.sizing_mandatory.rename({'sizing_storage': 'storage'})

            mand_ids = self.storage_capacity.coords['storage'].values[mandatory.values]
            if len(mand_ids):
                self.m.add_constraints(
                    self.storage_capacity.sel(storage=mand_ids) >= smin.sel(storage=mand_ids),
                    name='stor_invest_mand_lb',
                )

            opt_ids = self.storage_capacity.coords['storage'].values[~mandatory.values]
            if len(opt_ids):
                assert self.storage_capacity_indicator is not None
                smax = sds.sizing_max.rename({'sizing_storage': 'storage'})
                sc = self.storage_capacity.sel(storage=opt_ids)
                self.m.add_constraints(
                    sc >= smin.sel(storage=opt_ids) * self.storage_capacity_indicator, name='stor_invest_lb'
                )
                self.m.add_constraints(
                    sc <= smax.sel(storage=opt_ids) * self.storage_capacity_indicator, name='stor_invest_ub'
                )

    def _constrain_investment(self) -> None:
        """Constrain investment variables: build-once, active logic, size linking."""
        if self.invest_size is None:
            return

        fds = self.data.flows
        assert fds.invest_min is not None
        assert fds.invest_max is not None
        assert fds.invest_mandatory is not None
        assert fds.invest_lifetime is not None
        assert fds.invest_prior_size is not None
        assert self.invest_build is not None
        assert self.invest_active is not None
        assert self.invest_size_at_build is not None

        invest_ids = list(fds.invest_min.coords['invest_flow'].values)
        smin = fds.invest_min.rename({'invest_flow': 'flow'})
        smax = fds.invest_max.rename({'invest_flow': 'flow'})
        mandatory = fds.invest_mandatory.rename({'invest_flow': 'flow'})
        lifetime = fds.invest_lifetime.rename({'invest_flow': 'flow'})
        prior_size = fds.invest_prior_size.rename({'invest_flow': 'flow'})

        build = self.invest_build
        active = self.invest_active
        inv_size = self.invest_size
        assert self.flow_size is not None
        fs = self.flow_size.sel(flow=invest_ids)
        sab = self.invest_size_at_build

        periods = self.data.dims.period
        assert periods is not None
        period_vals = list(periods.values)
        n_periods = len(period_vals)

        # --- Build-once constraints ---
        build_sum = build.sum('period')
        mand_ids = [fid for fid, m in zip(invest_ids, mandatory.values, strict=True) if m]
        opt_ids = [fid for fid, m in zip(invest_ids, mandatory.values, strict=True) if not m]

        if mand_ids:
            self.m.add_constraints(build_sum.sel(flow=mand_ids) == 1, name='invest_build_once_mand')
        if opt_ids:
            self.m.add_constraints(build_sum.sel(flow=opt_ids) <= 1, name='invest_build_once_opt')

        # --- Active logic per flow ---
        for f_idx, fid in enumerate(invest_ids):
            lt = lifetime.values[f_idx]
            has_prior = prior_size.values[f_idx] > 0
            lt_int = int(lt) if not np.isnan(lt) else None

            for p_idx, p in enumerate(period_vals):
                b_sel = build.sel(flow=fid)
                a_sel = active.sel(flow=fid, period=p)

                if lt_int is None:
                    # No lifetime: once built, active forever
                    # active[p] = sum_{tau <= p} build[tau] + (1 if prior)
                    contributing = [period_vals[t] for t in range(p_idx + 1)]
                    rhs = b_sel.sel(period=contributing).sum('period')
                    if has_prior:
                        self.m.add_constraints(a_sel == rhs + 1, name=f'invest_active_{fid}_p{p}')
                    else:
                        self.m.add_constraints(a_sel == rhs, name=f'invest_active_{fid}_p{p}')
                else:
                    # With lifetime: active for lt_int periods after build
                    # active[p] = sum_{tau: p in [tau, tau+lt)} build[tau] + (1 if prior and p < lt)
                    contributing = [period_vals[t_idx] for t_idx in range(n_periods) if t_idx <= p_idx < t_idx + lt_int]
                    rhs = b_sel.sel(period=contributing).sum('period') if contributing else 0
                    if has_prior and p_idx < lt_int:
                        self.m.add_constraints(a_sel == rhs + 1, name=f'invest_active_{fid}_p{p}')
                    else:
                        self.m.add_constraints(a_sel == rhs, name=f'invest_active_{fid}_p{p}')

            # --- Prevent build when prior is active and no lifetime (can't build twice) ---
            if has_prior:
                self.m.add_constraints(build.sel(flow=fid).sum('period') == 0, name=f'invest_no_build_prior_{fid}')

        # --- Prior size: fix invest_size to prior_size ---
        prior_ids = [fid for fid, ps in zip(invest_ids, prior_size.values, strict=True) if ps > 0]
        if prior_ids:
            ps_vals = prior_size.sel(flow=prior_ids)
            self.m.add_constraints(inv_size.sel(flow=prior_ids) == ps_vals, name='invest_size_prior')

        # --- Size bounds ---
        # invest_size >= size_min (if mandatory or built)
        non_prior_mand = [fid for fid in mand_ids if fid not in prior_ids]
        if non_prior_mand:
            self.m.add_constraints(
                inv_size.sel(flow=non_prior_mand) >= smin.sel(flow=non_prior_mand),
                name='invest_size_lb_mand',
            )
        if opt_ids:
            non_prior_opt = [fid for fid in opt_ids if fid not in prior_ids]
            if non_prior_opt:
                # invest_size >= size_min * sum(build) — only if built
                self.m.add_constraints(
                    inv_size.sel(flow=non_prior_opt)
                    >= smin.sel(flow=non_prior_opt) * build_sum.sel(flow=non_prior_opt),
                    name='invest_size_lb_opt',
                )
                # invest_size <= size_max * sum(build) — zero when not built
                self.m.add_constraints(
                    inv_size.sel(flow=non_prior_opt)
                    <= smax.sel(flow=non_prior_opt) * build_sum.sel(flow=non_prior_opt),
                    name='invest_size_ub_opt',
                )

        # invest_size <= size_max (already via variable upper bound)

        # --- Size ↔ active linking (big-M: flow_size = invest_size when active, 0 when not) ---
        self.m.add_constraints(fs <= smax * active, name='invest_fs_ub_active')
        self.m.add_constraints(fs <= inv_size, name='invest_fs_ub_size')
        self.m.add_constraints(fs >= inv_size - smax * (1 - active), name='invest_fs_lb')

        # --- size_at_build: linearization of invest_size * build ---
        self.m.add_constraints(sab <= smax * build, name='invest_sab_ub_build')
        self.m.add_constraints(sab <= inv_size, name='invest_sab_ub_size')
        self.m.add_constraints(sab >= inv_size - smax * (1 - build), name='invest_sab_lb')

    def _constrain_status(self) -> None:
        """Add switch transition and duration tracking constraints for status flows."""
        if self.flow_on is None:
            return
        assert self.flow_startup is not None
        assert self.flow_shutdown is not None

        ds = self.data.flows
        assert ds.status_uptime_min is not None
        assert ds.status_uptime_max is not None
        assert ds.status_downtime_min is not None
        assert ds.status_downtime_max is not None
        assert ds.status_initial is not None

        # Rename status_flow -> flow to align with variable dims
        min_up = ds.status_uptime_min.rename({'status_flow': 'flow'})
        max_up = ds.status_uptime_max.rename({'status_flow': 'flow'})
        min_down = ds.status_downtime_min.rename({'status_flow': 'flow'})
        max_down = ds.status_downtime_max.rename({'status_flow': 'flow'})
        initial = ds.status_initial.rename({'status_flow': 'flow'})

        prev_up = (
            ds.status_previous_uptime.rename({'status_flow': 'flow'}) if ds.status_previous_uptime is not None else None
        )
        prev_down = (
            ds.status_previous_downtime.rename({'status_flow': 'flow'})
            if ds.status_previous_downtime is not None
            else None
        )

        # Filter to flows with known initial state
        has_initial = initial.notnull()
        previous_state = initial.sel(flow=initial.coords['flow'][has_initial]) if has_initial.any() else None

        add_switch_transitions(
            self.m,
            self.flow_on,
            self.flow_startup,
            self.flow_shutdown,
            name='status',
            previous_state=previous_state,
        )

        dt = self.data.dims.dt

        # Uptime tracking
        has_any_up = min_up.notnull().any() | max_up.notnull().any()
        if has_any_up:
            add_duration_tracking(
                self.m,
                self.flow_on,
                dt,
                name='uptime',
                minimum=min_up,
                maximum=max_up,
                previous=prev_up,
            )

        # Downtime tracking: state = 1 - on
        has_any_down = min_down.notnull().any() | max_down.notnull().any()
        if has_any_down:
            add_duration_tracking(
                self.m,
                1 - self.flow_on,
                dt,
                name='downtime',
                minimum=min_down,
                maximum=max_down,
                previous=prev_down,
            )

    def _constrain_component_status(self) -> None:
        """Add switch transition and duration tracking constraints for component status."""
        if self.component_on is None:
            return
        assert self.component_startup is not None
        assert self.component_shutdown is not None

        ds = self.data.flows
        assert ds.cstatus_uptime_min is not None
        assert ds.cstatus_uptime_max is not None
        assert ds.cstatus_downtime_min is not None
        assert ds.cstatus_downtime_max is not None
        assert ds.cstatus_initial is not None

        min_up = ds.cstatus_uptime_min.rename({'cstatus_component': 'component'})
        max_up = ds.cstatus_uptime_max.rename({'cstatus_component': 'component'})
        min_down = ds.cstatus_downtime_min.rename({'cstatus_component': 'component'})
        max_down = ds.cstatus_downtime_max.rename({'cstatus_component': 'component'})
        initial = ds.cstatus_initial.rename({'cstatus_component': 'component'})

        prev_up = (
            ds.cstatus_previous_uptime.rename({'cstatus_component': 'component'})
            if ds.cstatus_previous_uptime is not None
            else None
        )
        prev_down = (
            ds.cstatus_previous_downtime.rename({'cstatus_component': 'component'})
            if ds.cstatus_previous_downtime is not None
            else None
        )

        has_initial = initial.notnull()
        previous_state = initial.sel(component=initial.coords['component'][has_initial]) if has_initial.any() else None

        add_switch_transitions(
            self.m,
            self.component_on,
            self.component_startup,
            self.component_shutdown,
            name='cstatus',
            element_dim='component',
            previous_state=previous_state,
        )

        dt = self.data.dims.dt

        has_any_up = min_up.notnull().any() | max_up.notnull().any()
        if has_any_up:
            add_duration_tracking(
                self.m,
                self.component_on,
                dt,
                name='c_uptime',
                element_dim='component',
                minimum=min_up,
                maximum=max_up,
                previous=prev_up,
            )

        has_any_down = min_down.notnull().any() | max_down.notnull().any()
        if has_any_down:
            add_duration_tracking(
                self.m,
                1 - self.component_on,
                dt,
                name='c_downtime',
                element_dim='component',
                minimum=min_down,
                maximum=max_down,
                previous=prev_down,
            )

    def _create_balance(self) -> None:
        """Create carrier balance: ``sum_f(coeff * P) = 0`` for all carriers and timesteps."""
        d = self.data
        coeff = d.carriers.flow_coeff  # (carrier, flow) — NaN for unconnected

        # Replace NaN with 0 for summation (unconnected flows contribute nothing)
        coeff_filled = coeff.fillna(0)

        # Carrier balance: sum over flow dim of (coeff * flow_rate) == 0
        lhs = sparse_weighted_sum(self.flow_rate, coeff_filled, sum_dim='flow', group_dim='carrier')
        self.m.add_constraints(lhs == 0, name='carrier_balance')

    def _create_converter_constraints(self) -> None:
        """Create conversion constraints: ``sum_f(a_f * P) = 0`` per converter and equation."""
        d = self.data
        if d.converters is None:
            return

        ds = d.converters

        # Select flow rates for each (converter, flow) pair
        selected = self.flow_rate.sel(flow=ds.pair_flow)  # (pair, time)
        weighted = selected * ds.pair_coeff  # (pair, eq_idx, time)

        # Group by converter and sum over pairs
        mapping = xr.DataArray(ds.pair_converter.values, dims=['pair'], name='converter')
        lhs = weighted.groupby(mapping).sum()  # (converter, eq_idx, time)

        # Restore original converter order (groupby sorts alphabetically)
        conv_ids = list(ds.eq_mask.coords['converter'].values)
        lhs = lhs.sel(converter=conv_ids)

        # Drop flow coord left by vectorized sel
        lhs = lhs.drop_vars('flow', errors='ignore')

        # Broadcast eq_mask (converter, eq_idx) to (converter, eq_idx, time)
        mask_3d = ds.eq_mask.expand_dims(time=ds.pair_coeff.coords['time'])
        self.m.add_constraints(lhs == 0, name='conversion', mask=mask_3d)

    def _create_piecewise_constraints(self) -> None:
        """Create piecewise-linear conversion constraints via linopy's piecewise API.

        For each converter with ``PiecewiseConversion``: builds one
        ``add_piecewise_formulation`` call linking all curve flows through
        shared interpolation weights. The optional ``PiecewiseConversion.status``
        wires in the existing ``component_on`` binary as the ``active`` gate.
        Per-converter availability is enforced separately as
        ``flow_rate <= avail * max_bp * active``.
        """
        import warnings
        from typing import cast

        from linopy import EvolvingAPIWarning
        from linopy.piecewise import add_piecewise_formulation

        from fluxopt.types import PiecewiseMethod

        pw = self.data.piecewise
        if pw is None:
            return

        for conv_id in pw.converter_ids():
            method = cast('PiecewiseMethod', str(pw.method.sel(pw_converter=conv_id).values))
            avail = pw.availability.sel(pw_converter=conv_id)  # (time,)
            has_status = bool(pw.has_status.sel(pw_converter=conv_id).values)

            active = self.component_on.sel(component=conv_id) if has_status and self.component_on is not None else None

            mask = pw.pair_converter.values == conv_id
            pair_indices = np.where(mask)[0]

            pairs: list[tuple[Any, ...]] = []
            for idx in pair_indices:
                fid = str(pw.pair_flow.values[idx])
                bound = str(pw.pair_bound.values[idx])
                # Wrap as LinearExpression and drop the per-flow scalar coord so
                # linopy can broadcast pairs without merge conflicts on 'flow'.
                expr = (1.0 * self.flow_rate.sel(flow=fid)).drop_vars('flow', errors='ignore')
                bps = (
                    pw.breakpoints.isel(pw_pair=idx)
                    .drop_vars('pw_pair', errors='ignore')
                    .rename({'breakpoint': '_breakpoint'})
                )
                pairs.append((expr, bps) if bound == '==' else (expr, bps, bound))

            # fluxopt owns the API risk of add_piecewise_formulation — surface it
            # in our changelog instead of leaking warnings into user output.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', EvolvingAPIWarning)
                formulation = add_piecewise_formulation(
                    self.m,
                    *pairs,
                    method=method,
                    active=active,
                    name=f'pw_{conv_id}',
                )
            self._piecewise[conv_id] = formulation

            # Availability constraint: scale upper envelope, not the curve.
            # Use max-over-breakpoint, not last — SOS2 allows non-monotonic breakpoints.
            ref_expr = pairs[0][0]
            ref_idx = pair_indices[0]
            max_bp = pw.breakpoints.isel(pw_pair=ref_idx).max('breakpoint')  # (time,)
            if active is not None:
                self.m.add_constraints(
                    ref_expr <= avail * max_bp * active,
                    name=f'pw_avail_{conv_id}',
                )
            else:
                self.m.add_constraints(
                    ref_expr <= avail * max_bp,
                    name=f'pw_avail_{conv_id}',
                )

    def _create_effects(self) -> None:
        """Effect tracking: temporal and lump domains."""
        d = self.data
        ds = d.effects

        effect_ids = ds.total_min.coords['effect']

        if len(effect_ids) == 0:
            return

        # --- Temporal domain: effect_temporal[effect, time(, period)] ---
        self.effect_temporal = self.m.add_variables(
            coords={'effect': effect_ids, **self.data.dims.coords(time=True, period=True)},
            name='effect--temporal',
        )

        # Flow contributions: sum_f(coeff_{f,k,t} * P_{f,t} * dt_t)
        effect_coeff = d.flows.effect_coeff  # (flow, effect, time)
        has_any_coeff = (effect_coeff != 0).any()

        temporal_rhs: Any = 0
        if has_any_coeff:
            coeff_dt = effect_coeff * d.dims.dt
            temporal_rhs = sparse_weighted_sum(self.flow_rate, coeff_dt, sum_dim='flow', group_dim='effect')

        # Status running costs: sum_f(running_coeff[f,k,t] * on[f,t] * dt[t])
        if d.flows.status_effects_running is not None:
            assert self.flow_on is not None
            er = d.flows.status_effects_running.rename({'status_flow': 'flow'})
            if (er != 0).any():
                temporal_rhs = temporal_rhs + (er * self.flow_on * d.dims.dt).sum('flow')

        # Status startup costs: sum_f(startup_coeff[f,k,t] * startup[f,t]) — per event, no dt
        if d.flows.status_effects_startup is not None:
            assert self.flow_startup is not None
            es = d.flows.status_effects_startup.rename({'status_flow': 'flow'})
            if (es != 0).any():
                temporal_rhs = temporal_rhs + (es * self.flow_startup).sum('flow')

        # Component-level status running costs
        if d.flows.cstatus_effects_running is not None:
            assert self.component_on is not None
            cer = d.flows.cstatus_effects_running.rename({'cstatus_component': 'component'})
            if (cer != 0).any():
                temporal_rhs = temporal_rhs + (cer * self.component_on * d.dims.dt).sum('component')

        # Component-level status startup costs
        if d.flows.cstatus_effects_startup is not None:
            assert self.component_startup is not None
            ces = d.flows.cstatus_effects_startup.rename({'cstatus_component': 'component'})
            if (ces != 0).any():
                temporal_rhs = temporal_rhs + (ces * self.component_startup).sum('component')

        # Cross-effect temporal: cf_temporal[k,j,t] * effect_temporal[j,t]
        if ds.cf_temporal is not None:
            source_t = self.effect_temporal.rename({'effect': 'source_effect'})
            temporal_rhs = temporal_rhs + (ds.cf_temporal * source_t).sum('source_effect')

        self.m.add_constraints(self.effect_temporal == temporal_rhs, name='effect_temporal_eq')

        # Per-hour bounds: effect[t] <= rate_max * dt[t]
        # effect_temporal is in absolute units (e.g. EUR), so the per-hour rate
        # must be scaled by the timestep duration to get the per-timestep limit.
        dt = d.dims.dt
        min_ph = ds.rate_min * dt  # (effect, time) — NaN = unbounded
        has_min_ph = min_ph.notnull()
        if has_min_ph.any():
            self.m.add_constraints(self.effect_temporal >= min_ph, name='effect_min_ph', mask=has_min_ph)

        max_ph = ds.rate_max * dt
        has_max_ph = max_ph.notnull()
        if has_max_ph.any():
            self.m.add_constraints(self.effect_temporal <= max_ph, name='effect_max_ph', mask=has_max_ph)

        # --- Lump domain: effect_lump[effect(, period)] ---
        # Combines all non-temporal contributions (sizing, investment recurring, investment at-build)
        pc = self.data.dims.coords(period=True)
        self.effect_lump = self.m.add_variables(coords={'effect': effect_ids, **pc}, name='effect--lump')

        # Accumulate direct lump contributions per effect
        lump_direct: Any = 0

        # Flow sizing: per-size costs (Sizing only, not Investment)
        if d.flows.sizing_effects_per_size is not None:
            sizing_ids = list(d.flows.sizing_effects_per_size.coords['sizing_flow'].values)
            eps = d.flows.sizing_effects_per_size.rename({'sizing_flow': 'flow'})
            if (eps != 0).any():
                assert self.flow_size is not None
                lump_direct = lump_direct + (eps * self.flow_size.sel(flow=sizing_ids)).sum('flow')

        # Flow sizing: fixed costs — optional (binary * cost), mandatory (constant)
        if self.flow_size_indicator is not None:
            assert d.flows.sizing_effects_fixed is not None
            opt_ids = list(self.flow_size_indicator.coords['flow'].values)
            ef = d.flows.sizing_effects_fixed.rename({'sizing_flow': 'flow'}).sel(flow=opt_ids)
            if (ef != 0).any():
                lump_direct = lump_direct + (ef * self.flow_size_indicator).sum('flow')
        if (
            self.flow_size is not None
            and d.flows.sizing_effects_fixed is not None
            and d.flows.sizing_mandatory is not None
        ):
            mand_mask = d.flows.sizing_mandatory.values
            if mand_mask.any():
                mand_ids = list(d.flows.sizing_mandatory.coords['sizing_flow'].values[mand_mask])
                ef_mand = d.flows.sizing_effects_fixed.sel(sizing_flow=mand_ids)
                if (ef_mand != 0).any():
                    lump_direct = lump_direct + ef_mand.sum('sizing_flow')

        # Storage sizing: per-size costs
        if (
            self.storage_capacity is not None
            and d.storages is not None
            and d.storages.sizing_effects_per_size is not None
        ):
            eps = d.storages.sizing_effects_per_size.rename({'sizing_storage': 'storage'})
            if (eps != 0).any():
                lump_direct = lump_direct + (eps * self.storage_capacity).sum('storage')

        # Storage sizing: fixed costs — optional (binary * cost), mandatory (constant)
        if (
            self.storage_capacity_indicator is not None
            and d.storages is not None
            and d.storages.sizing_effects_fixed is not None
        ):
            opt_ids = list(self.storage_capacity_indicator.coords['storage'].values)
            ef = d.storages.sizing_effects_fixed.rename({'sizing_storage': 'storage'}).sel(storage=opt_ids)
            if (ef != 0).any():
                lump_direct = lump_direct + (ef * self.storage_capacity_indicator).sum('storage')
        if (
            self.storage_capacity is not None
            and d.storages is not None
            and d.storages.sizing_effects_fixed is not None
            and d.storages.sizing_mandatory is not None
        ):
            mand_mask = d.storages.sizing_mandatory.values
            if mand_mask.any():
                mand_ids = list(d.storages.sizing_mandatory.coords['sizing_storage'].values[mand_mask])
                ef_mand = d.storages.sizing_effects_fixed.sel(sizing_storage=mand_ids)
                if (ef_mand != 0).any():
                    lump_direct = lump_direct + ef_mand.sum('sizing_storage')

        # Investment: recurring per-size costs
        if self.invest_active is not None and d.flows.invest_effects_per_size_recurring is not None:
            eps_p = d.flows.invest_effects_per_size_recurring.rename({'invest_flow': 'flow'})
            if (eps_p != 0).any():
                assert self.flow_size is not None
                invest_ids = list(d.flows.invest_effects_per_size_recurring.coords['invest_flow'].values)
                lump_direct = lump_direct + (eps_p * self.flow_size.sel(flow=invest_ids)).sum('flow')

        # Investment: recurring fixed costs
        if self.invest_active is not None and d.flows.invest_effects_fixed_recurring is not None:
            ef_p = d.flows.invest_effects_fixed_recurring.rename({'invest_flow': 'flow'})
            if (ef_p != 0).any():
                lump_direct = lump_direct + (ef_p * self.invest_active).sum('flow')

        # Investment: at-build per-size costs (charged in build period)
        if self.invest_size_at_build is not None and d.flows.invest_effects_per_size_at_build is not None:
            eps_once = d.flows.invest_effects_per_size_at_build.rename({'invest_flow': 'flow'})
            if (eps_once != 0).any():
                lump_direct = lump_direct + (eps_once * self.invest_size_at_build).sum('flow')

        # Investment: at-build fixed costs (charged in build period)
        if self.invest_build is not None and d.flows.invest_effects_fixed_at_build is not None:
            ef_once = d.flows.invest_effects_fixed_at_build.rename({'invest_flow': 'flow'})
            if (ef_once != 0).any():
                lump_direct = lump_direct + (ef_once * self.invest_build).sum('flow')

        # Cross-effect lump: mean(cf_temporal, 'time')[k,j] * effect_lump[j]
        # Time-varying contribution_from values are averaged over time for the
        # lump domain. Warn per-(k,j) where the factor varies and the mean
        # is non-zero (i.e. the cross-effect actually contributes).
        lump_rhs: Any = lump_direct
        if ds.cf_temporal is not None:
            cf_lump = ds.cf_temporal.mean('time')
            varying = (ds.cf_temporal != cf_lump).any('time')  # (effect, source_effect)
            non_trivial = varying & (cf_lump != 0)
            if bool(non_trivial.any().item()) and not isinstance(lump_direct, int):
                import warnings

                pairs = [
                    (str(non_trivial.coords['effect'].values[i]), str(non_trivial.coords['source_effect'].values[j]))
                    for i, j in zip(*non_trivial.values.nonzero(), strict=True)
                ]
                pair_str = ', '.join(f'{k}<-{j}' for k, j in pairs)
                warnings.warn(
                    f'Time-varying contribution_from for {pair_str} is averaged over time for the lump domain. '
                    "If this isn't what you want, split into separate effects.",
                    stacklevel=2,
                )
            source_p = self.effect_lump.rename({'effect': 'source_effect'})
            cross = (cf_lump * source_p).sum('source_effect')
            lump_rhs = cross + lump_direct  # linopy expr must be left operand

        self.m.add_constraints(self.effect_lump == lump_rhs, name='effect_lump_eq')

        # --- Total: effect_total[effect(, period)] ---
        self.effect_total = self.m.add_variables(coords={'effect': effect_ids, **pc}, name='effect--total')
        temporal_sum = (self.effect_temporal * d.dims.weights).sum('time')
        rhs = temporal_sum + self.effect_lump
        self.m.add_constraints(self.effect_total == rhs, name='effect_total_eq')

        # Per-period bounds on effect_total
        min_pp = ds.periodic_min  # (effect[, period]) — NaN = unbounded
        max_pp = ds.periodic_max
        has_min_pp = min_pp.notnull()
        if has_min_pp.any():
            self.m.add_constraints(self.effect_total >= min_pp, name='effect_periodic_min', mask=has_min_pp)
        has_max_pp = max_pp.notnull()
        if has_max_pp.any():
            self.m.add_constraints(self.effect_total <= max_pp, name='effect_periodic_max', mask=has_max_pp)

        # Weighted total bounds (across all periods)
        # Single-period: effect_total has no period dim, bound applies directly.
        # Multi-period: weighted sum across periods, using per-effect period_weights
        # if set, else global period_weights, else unweighted.
        total_min = ds.total_min  # (effect,) — NaN = unbounded
        total_max = ds.total_max
        total_sum: Any
        if 'period' in self.effect_total.dims:
            # Multi-period: weighted sum across periods.
            # Per-effect weights override global; both are always set in multi-period.
            assert d.dims.period_weights is not None
            if ds.period_weights is not None:
                w_per_effect = ds.period_weights.fillna(d.dims.period_weights)
                total_sum = (self.effect_total * w_per_effect).sum('period')
            else:
                total_sum = (self.effect_total * d.dims.period_weights).sum('period')
        else:
            total_sum = self.effect_total
        has_min = total_min.notnull()
        if has_min.any():
            self.m.add_constraints(total_sum >= total_min, name='effect_total_min', mask=has_min)
        has_max = total_max.notnull()
        if has_max.any():
            self.m.add_constraints(total_sum <= total_max, name='effect_total_max', mask=has_max)

    def _create_storage(self) -> None:
        """Create storage variables, level balance, and prior/cyclic conditions."""
        d = self.data
        if d.storages is None:
            return
        ds = d.storages

        stor_ids = ds.capacity.coords['storage']

        # storage_level[storage, time(, period)] >= 0  (end-of-period convention)
        self.storage_level = self.m.add_variables(
            lower=0,
            coords={'storage': stor_ids, **self.data.dims.coords(time=True, period=True)},
            name='storage--level',
        )

        # --- Capacity bounds on storage_level ---
        cap = ds.capacity  # (storage,) — NaN for uncapped/investable
        has_fixed_cap = cap.notnull()
        has_invest_cap = self.storage_capacity is not None

        # Fixed-capacity storages: level <= capacity (parameter)
        if has_fixed_cap.any():
            cap_2d = cap.broadcast_like(xr.DataArray(np.nan, dims=['storage', 'time'], coords=[stor_ids, d.dims.time]))
            mask_cap = has_fixed_cap.broadcast_like(cap_2d)
            self.m.add_constraints(self.storage_level <= cap_2d, name='level_cap', mask=mask_cap)

        # Investable storages: level <= capacity (variable)
        if has_invest_cap:
            assert self.storage_capacity is not None
            invest_ids = list(self.storage_capacity.coords['storage'].values)
            level_invest = self.storage_level.sel(storage=invest_ids)
            self.m.add_constraints(level_invest <= self.storage_capacity, name='level_cap_invest')

        # --- Relative level bounds ---
        # For fixed-capacity storages
        if has_fixed_cap.any():
            rel_lb = ds.rel_level_lb
            rel_ub = ds.rel_level_ub

            abs_lb = rel_lb * cap
            has_lb = has_fixed_cap.broadcast_like(rel_lb) & (abs_lb > 1e-12)
            if has_lb.any():
                self.m.add_constraints(self.storage_level >= abs_lb, name='level_lb', mask=has_lb)

            abs_ub = rel_ub * cap
            cap_2d_check = cap.broadcast_like(rel_ub)
            has_ub = has_fixed_cap.broadcast_like(rel_ub) & (abs_ub < cap_2d_check - 1e-12)
            if has_ub.any():
                self.m.add_constraints(self.storage_level <= abs_ub, name='level_ub', mask=has_ub)

        # For investable storages: relative bounds use capacity variable
        if has_invest_cap:
            assert self.storage_capacity is not None
            invest_ids = list(self.storage_capacity.coords['storage'].values)
            rel_lb_inv = ds.rel_level_lb.sel(storage=invest_ids)
            rel_ub_inv = ds.rel_level_ub.sel(storage=invest_ids)
            level_invest = self.storage_level.sel(storage=invest_ids)

            has_lb = (rel_lb_inv > 1e-12).any('time')
            if has_lb.any():
                lb_mask = has_lb.broadcast_like(rel_lb_inv) & (rel_lb_inv > 1e-12)
                self.m.add_constraints(
                    level_invest >= rel_lb_inv * self.storage_capacity,
                    name='level_lb_invest',
                    mask=lb_mask,
                )

            has_ub = (rel_ub_inv < 1 - 1e-12).any('time')
            if has_ub.any():
                ub_mask = has_ub.broadcast_like(rel_ub_inv) & (rel_ub_inv < 1 - 1e-12)
                self.m.add_constraints(
                    level_invest <= rel_ub_inv * self.storage_capacity,
                    name='level_ub_invest',
                    mask=ub_mask,
                )

        # Map charge/discharge flows to storage dimension via sel + rename
        charge_fids = [str(v) for v in ds.charge_flow.values]
        discharge_fids = [str(v) for v in ds.discharge_flow.values]
        stor_vals = stor_ids.values

        charge_rates = self.flow_rate.sel(flow=charge_fids).rename({'flow': 'storage'})
        charge_rates = charge_rates.assign_coords({'storage': stor_vals})
        discharge_rates = self.flow_rate.sel(flow=discharge_fids).rename({'flow': 'storage'})
        discharge_rates = discharge_rates.assign_coords({'storage': stor_vals})

        # Precompute pure-xarray coefficients (no linopy overhead)
        loss_factor = (1 - ds.loss) ** d.dims.dt  # (storage, time)
        charge_factor = ds.eta_c * d.dims.dt  # (storage, time)
        discharge_factor = d.dims.dt / ds.eta_d  # (storage, time)

        inflow = charge_rates * charge_factor
        outflow = discharge_rates * discharge_factor

        # --- Prior variable + single balance for ALL storages ---
        # prior[storage] is a free variable representing the state before period 0.
        # Cyclic and fixed-prior constraints pin it; otherwise it's free.
        self.prior_storage_level = self.m.add_variables(
            lower=0, coords={'storage': stor_ids, **self.data.dims.coords(period=True)}, name='storage--prior'
        )

        add_accumulation_constraints(
            self.m,
            self.storage_level,
            inflow=inflow,
            outflow=outflow,
            decay=loss_factor,
            initial=self.prior_storage_level,
            name='storage_balance',
        )

        # Cyclic: prior == level[-1]
        cyclic_mask = ds.cyclic.values.astype(bool)
        if np.any(cyclic_mask):
            cyc_ids = [str(s) for s, c in zip(stor_ids.values, cyclic_mask, strict=True) if c]
            self.m.add_constraints(
                self.prior_storage_level.sel(storage=cyc_ids) == self.storage_level.sel(storage=cyc_ids).isel(time=-1),
                name='storage_prior_cyc',
            )

        # Fixed prior: prior == value
        has_prior = ds.prior_level.notnull().values
        if np.any(has_prior):
            prior_ids = [str(s) for s, p in zip(stor_ids.values, has_prior, strict=True) if p]
            self.m.add_constraints(
                self.prior_storage_level.sel(storage=prior_ids) == ds.prior_level.sel(storage=prior_ids),
                name='storage_prior_fix',
            )

    def _set_objective(self) -> None:
        """Set objective: minimize the sum of (period-weighted) effect totals.

        Objective = sum_k sum_p( ω[k,p] * effect_total[k,p] )

        ω falls back to global period_weights (or 1 in single-period).
        """
        ds = self.data.effects
        obj_expr: Any = 0

        for k in self._objective_effects:
            effect_ids = list(ds.total_min.coords['effect'].values)
            if k not in effect_ids:
                raise ValueError(f'Objective effect {k!r} not found. Available: {effect_ids}')

            # Resolve per-effect weight, falling back to global period_weights, then 1
            w: xr.DataArray | int = 1
            if ds.period_weights is not None and not ds.period_weights.sel(effect=k).isnull().all():
                w = ds.period_weights.sel(effect=k)
            elif self.data.dims.period_weights is not None:
                w = self.data.dims.period_weights

            obj_expr = obj_expr + (w * self.effect_total.sel(effect=k)).sum()

        self.m.add_objective(obj_expr)
