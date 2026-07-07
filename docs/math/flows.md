# Flows

A flow represents energy transfer on a bus. Each flow \(f\) has a non-negative
rate variable \(P_{f,t}\) bounded by an optional nominal capacity \(\bar{\mathrm{P}}_f\).

## Bounds

When `Flow.size` is set, the rate is bounded by relative minimum and maximum
profiles:

\[
\bar{\mathrm{P}}_f \cdot \underline{\mathrm{p}}_{f,t} \leq P_{f,t} \leq \bar{\mathrm{P}}_f \cdot \bar{\mathrm{p}}_{f,t} \quad \forall \, f, t
\]

By default \(\underline{\mathrm{p}}_{f,t} = 0\) and \(\bar{\mathrm{p}}_{f,t} = 1\), so the bounds
simplify to \(0 \leq P_{f,t} \leq \bar{\mathrm{P}}_f\). With no capacity
(\(\bar{\mathrm{P}}_f = \infty\)) the flow is just \(P_{f,t} \geq 0\).

## Fixed Profile

When `Flow.fixed_relative_profile` (\(\pi_{f,t}\)) is set, the rate is pinned
to a profile scaled by the capacity:

\[
P_{f,t} = \bar{\mathrm{P}}_f \cdot \pi_{f,t} \quad \forall \, f, t
\]

Implemented by setting both bounds equal to the profile value.

## Aggregate Bounds

Aggregate bounds constrain the flow-hours \(H_{f,p}\) — the flow rate summed
over the horizon — within **each period independently**:

\[
H_{f,p} = \sum_{t \in \mathcal{T}} P_{f,t,p} \cdot \Delta t_t
\]

**Flow-hour bounds** (`flow_hours_min` / `flow_hours_max`) — absolute [MWh]:

\[
\underline{\mathrm{H}}_f \leq H_{f,p} \leq \bar{\mathrm{H}}_f \quad \forall \, p
\]

**Load factor bounds** (`load_factor_min` / `load_factor_max`) — utilization
relative to capacity, with \(T = \sum_t \Delta t_t\) the period duration:

\[
\underline{\lambda}_f \cdot \bar{\mathrm{P}}_f \cdot T \leq H_{f,p} \leq \bar{\lambda}_f \cdot \bar{\mathrm{P}}_f \cdot T \quad \forall \, p
\]

When `Flow.size` is a [Sizing](sizing.md) or Investment object,
\(\bar{\mathrm{P}}_f\) is the size *variable* \(S_{f,p}\) and the products
stay linear (constant \(\lambda \cdot T\) times a variable). Load factor
bounds therefore require a size; flow-hour bounds do not.

Cross-period budgets are not expressed here — route them through an
[effect](effects.md) with `total_min` / `total_max`.

## Effect Contributions

Each flow contributes to tracked effects (cost, emissions, …). Per-timestep:

\[
\mathrm{c}_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t
\]

\(\mathrm{c}_{f,k,t}\) is the per-flow-hour coefficient (`Flow.effects_per_flow_hour`).
Units cancel: e.g. €/MWh × MW × h = €. Contributions feed into the
[effect tracking](effects.md) equations.

## See also

- [Sizing](sizing.md) — when `Flow.size` is a `Sizing` object instead of a fixed value.
- [Status](status.md) — when `Flow.status` is set, the flow becomes semi-continuous \(\{0\} \cup [\underline{\mathrm{P}}, \bar{\mathrm{P}}]\).
- [Carrier Balance — Multi-Node](carrier-balance.md#multi-node-carriers) — when `Flow.node` targets a specific node.

## Parameters

| Symbol | Description | API |
|---|---|---|
| \(P_{f,t}\) | Flow rate variable | `flow--rate[flow, time]` |
| \(\bar{\mathrm{P}}_f\) | Nominal capacity | [`Flow.size`](../api/fluxopt/elements.md#fluxopt.elements.Flow(size)) |
| \(\underline{\mathrm{p}}_{f,t}\) | Relative lower bound | [`Flow.relative_rate_min`](../api/fluxopt/elements.md#fluxopt.elements.Flow(relative_rate_min)) |
| \(\bar{\mathrm{p}}_{f,t}\) | Relative upper bound | [`Flow.relative_rate_max`](../api/fluxopt/elements.md#fluxopt.elements.Flow(relative_rate_max)) |
| \(\pi_{f,t}\) | Fixed relative profile | [`Flow.fixed_relative_profile`](../api/fluxopt/elements.md#fluxopt.elements.Flow(fixed_relative_profile)) |
| \(\mathrm{c}_{f,k,t}\) | Effect coefficient per flow-hour | [`Flow.effects_per_flow_hour`](../api/fluxopt/elements.md#fluxopt.elements.Flow(effects_per_flow_hour)) |
| \(\underline{\mathrm{H}}_f\) | Minimum flow-hours per period | [`Flow.flow_hours_min`](../api/fluxopt/elements.md#fluxopt.elements.Flow(flow_hours_min)) |
| \(\bar{\mathrm{H}}_f\) | Maximum flow-hours per period | [`Flow.flow_hours_max`](../api/fluxopt/elements.md#fluxopt.elements.Flow(flow_hours_max)) |
| \(\underline{\lambda}_f\) | Minimum load factor per period | [`Flow.load_factor_min`](../api/fluxopt/elements.md#fluxopt.elements.Flow(load_factor_min)) |
| \(\bar{\lambda}_f\) | Maximum load factor per period | [`Flow.load_factor_max`](../api/fluxopt/elements.md#fluxopt.elements.Flow(load_factor_max)) |
| \(\Delta t_t\) | Timestep duration (h) | dt |

See [Notation](notation.md) for the full symbol table and [Indexing
Convention](notation.md#indexing-convention) for how indices broadcast.
