# Sizing (Investment Optimization)

## Overview

Sizing introduces a capacity decision variable \(S\) that replaces the fixed
nominal capacity \(\bar{\mathrm{P}}_f\). The solver optimizes both the capacity and the
dispatch simultaneously.

## Variables

| Symbol | Code | Domain | Description |
|---|---|---|---|
| \(S_f\) | `flow--size[flow]` | \(\geq 0\) | Invested flow capacity |
| \(y_f\) | `flow--size_indicator[flow]` | \(\{0, 1\}\) | Binary: invest yes/no (optional only) |
| \(S_s\) | `storage--capacity[storage]` | \(\geq 0\) | Invested storage capacity |
| \(y_s\) | `storage--size_indicator[storage]` | \(\{0, 1\}\) | Binary: invest yes/no (optional only) |

## Mandatory Sizing

When `mandatory=True`, the component must be built. The capacity is continuous:

\[
\mathrm{S}^- \leq S_f \leq \mathrm{S}^+ \quad \text{(mandatory)}
\]

where \(\mathrm{S}^-\) = `min_size` and \(\mathrm{S}^+\) = `max_size`. No binary variable is needed,
so the problem is faster to solve:

## Optional Sizing

When `mandatory=False`, a binary indicator \(y_f\) gates the capacity:

\[
\mathrm{S}^- \cdot y_f \leq S_f \leq \mathrm{S}^+ \cdot y_f
\]

When \(y_f = 0\): \(S_f = 0\) (not built). When \(y_f = 1\): \(S_f \in [\mathrm{S}^-, \mathrm{S}^+]\).
Use this when you need `effects_fixed` (one-time costs gated by the indicator)
or when `min_size > 0` must be enforced only if built:

### Binary Invest

When \(\mathrm{S}^- = \mathrm{S}^+\), the sizing reduces to a binary yes/no decision at exactly
that capacity:

## Flow Rate Bounds with Sizing

With sizing, the fixed capacity \(\bar{\mathrm{P}}_f\) is replaced by the variable \(S_f\).
The relative bounds scale by the invested size:

\[
S_f \cdot \underline{\mathrm{p}}_{f,t} \leq P_{f,t} \leq S_f \cdot \bar{\mathrm{p}}_{f,t} \quad \forall \, t
\]

Similarly for fixed profiles:

\[
P_{f,t} = S_f \cdot \pi_{f,t} \quad \forall \, t
\]

## Storage Sizing

The same pattern applies to storage capacity. The charge state bounds become:

\[
S_s \cdot \underline{\mathrm{e}}_s \leq E_{s,t} \leq S_s \cdot \bar{\mathrm{e}}_s \quad \forall \, t
\]

## Investment Effects

Investment costs contribute to effect totals. For each effect \(k\):

### Per-Size

Cost proportional to the invested size (e.g. €/MW):

\[
\Phi_k^{\text{invest,perSize}} = \sum_{f} \gamma_{f,k} \cdot S_f + \sum_{s} \gamma_{s,k} \cdot S_s
\]

where \(\gamma_{f,k}\) is `Sizing.effects_per_size[k]`.

### Fixed

One-time cost charged when the component is built, gated by the binary indicator:

\[
\Phi_k^{\text{invest,fixed}} = \sum_{f} \phi_{f,k} \cdot y_f + \sum_{s} \phi_{s,k} \cdot y_s
\]

where \(\phi_{f,k}\) is `Sizing.effects_fixed[k]`. Only applies when
`mandatory=False` (binary indicator exists).

### Total

The direct investment contribution to effect \(k\) is:

\[
\Phi_k^{\text{invest}} = \Phi_k^{\text{invest,perSize}} + \Phi_k^{\text{invest,fixed}}
\]

This feeds into the [effect total](effects.md) equation and can be further
weighted by [cross-effect contributions](effects.md#cross-effect-contributions).

## Interaction with Other Features

### With Bounds

Relative bounds (`relative_minimum`, `relative_maximum`) are fractions of the
**optimized** size variable, not a fixed number. If the solver picks 80 MW and
`relative_minimum=0.3`, the minimum flow rate is 24 MW.

### With Status

When a flow has both `Sizing` and `Status`, a big-M formulation decouples the
binary on/off from the continuous size. See [Status — Interaction with Sizing](status.md#interaction-with-sizing)
for the constraints.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(S_f\) | Flow capacity variable | `flow--size[flow]` |
| \(S_s\) | Storage capacity variable | `storage--capacity[storage]` |
| \(y_f\), \(y_s\) | Binary invest indicator | `flow--size_indicator`, `storage--size_indicator` |
| \(\mathrm{S}^-\) | Minimum size | [`Sizing.min_size`](../api/fluxopt/elements.md#fluxopt.elements.Sizing(min_size)) |
| \(\mathrm{S}^+\) | Maximum size | [`Sizing.max_size`](../api/fluxopt/elements.md#fluxopt.elements.Sizing(max_size)) |
| \(\gamma_{f,k}\) | Per-size investment cost | [`Sizing.effects_per_size`](../api/fluxopt/elements.md#fluxopt.elements.Sizing(effects_per_size)) |
| \(\phi_{f,k}\) | Fixed investment cost | [`Sizing.effects_fixed`](../api/fluxopt/elements.md#fluxopt.elements.Sizing(effects_fixed)) |

See [Notation](notation.md) for the full symbol table.
