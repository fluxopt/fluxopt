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
| \(\underline{\mathrm{p}}_{f,t}\) | Relative lower bound | [`Flow.relative_minimum`](../api/fluxopt/elements.md#fluxopt.elements.Flow(relative_minimum)) |
| \(\bar{\mathrm{p}}_{f,t}\) | Relative upper bound | [`Flow.relative_maximum`](../api/fluxopt/elements.md#fluxopt.elements.Flow(relative_maximum)) |
| \(\pi_{f,t}\) | Fixed relative profile | [`Flow.fixed_relative_profile`](../api/fluxopt/elements.md#fluxopt.elements.Flow(fixed_relative_profile)) |
| \(\mathrm{c}_{f,k,t}\) | Effect coefficient per flow-hour | [`Flow.effects_per_flow_hour`](../api/fluxopt/elements.md#fluxopt.elements.Flow(effects_per_flow_hour)) |
| \(\Delta t_t\) | Timestep duration (h) | dt |

See [Notation](notation.md) for the full symbol table and [Indexing
Convention](notation.md#indexing-convention) for how indices broadcast.
