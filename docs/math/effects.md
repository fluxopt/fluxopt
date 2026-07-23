# Effect Tracking & Bounding

## Overview

Effects represent quantities that are tracked across the optimization horizon (e.g.,
cost, CO₂ emissions, primary energy). One effect is designated as the objective to
minimize.

Effects are split into two **domains** based on how they vary over time and
how they are weighted in multi-period optimization:

| Domain | Dims | What goes here | Multi-period weighting |
|---|---|---|---|
| **Temporal** | `(effect, time)` | Flow costs, running costs, startup costs — anything that varies per timestep | Summed over time (× \(\mathrm{w}_t\)), then weighted by `period_weights` |
| **Lump** | `(effect,)` | Sizing costs, fixed O&M, one-time CAPEX — anything not per-timestep | Weighted by \(\omega_{k,p}\) (defaults to global `period_weights`) |

All domains support cross-effect chains via `contribution_from`.

In multi-period mode, all variables gain an optional `period` dimension.
See [Objective](objective.md) for how the domains are weighted in the objective.

## Temporal Domain

Each effect accumulates contributions from all flows at each timestep:

\[
\Phi_{k,t}^{\text{temporal}} = \underbrace{\sum_{f \in \mathcal{F}} \mathrm{c}_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t}_{\text{direct flow contributions}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha_{k,j,t} \cdot \Phi_{j,t}^{\text{temporal}}}_{\text{cross-effect contributions}} \quad \forall \, k, t
\]

The coefficient \(\mathrm{c}_{f,k,t}\) specifies how much of effect \(k\) is produced per
flow-hour of flow \(f\) (e.g., €/MWh for cost, kg/MWh for emissions).

The cross-effect factor \(\alpha_{k,j,t}\) can be time-varying or constant
(both via `contribution_from`).

\(\Phi_{k,t}^{\text{temporal}}\) is an **expression**, not a solver variable:
no per-timestep effect variables exist in the model. The recursive definition
above has the closed form

\[
\boldsymbol{\Phi}_t^{\text{temporal}} = (I - A_t)^{-1} \, \boldsymbol{D}_t
\]

where \(A_t = [\alpha_{k,j,t}]\) and \(\boldsymbol{D}_t\) collects the direct
contributions — the Leontief inverse is computed numerically at build time and
multi-level chains (e.g., PE → CO₂ → cost) substitute inline. The expression is
summed over time directly into the total (see below); per-timestep effect
series in results are reconstructed post-solve from flow rates and
coefficients.

## Lump Domain

Sizing costs, fixed costs, and one-time costs (not time-varying) are accumulated per effect:

\[
\Phi_k^{\text{lump}} = \underbrace{\Phi_k^{\text{invest,direct}}}_{\text{direct sizing costs}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha_{k,j} \cdot \Phi_j^{\text{lump}}}_{\text{cross-effect contributions}} \quad \forall \, k
\]

where the direct investment term is:

\[
\Phi_k^{\text{invest,direct}} = \sum_{f} \gamma_{f,k} \cdot S_f + \sum_{f} \phi_{f,k} \cdot y_f + \sum_{s} \gamma_{s,k} \cdot S_s + \sum_{s} \phi_{s,k} \cdot y_s
\]

Because \(\Phi_k^{\text{lump}}\) is a **variable** (not an expression), the
solver resolves multi-level chains correctly: if PE has sizing costs
and CO₂ depends on PE and cost depends on CO₂, the chain propagates through the
lump domain just as it does through the temporal domain.

## Cross-Effect Contributions

An effect can include a weighted fraction of another effect's value via
`contribution_from`. This enables patterns like carbon pricing (CO₂ → cost)
or transitive chains (PE → CO₂ → cost).

The factor \(\alpha_{k,j}\) from `contribution_from` accepts either a scalar
or a `Variate`:

- **Scalar**: applied identically to both temporal and lump domains.
- **Variate** (time-varying): applied per-timestep in the temporal domain.
  Rejected at build time when the source effect carries lump (sizing/fixed)
  contributions — a per-timestep factor has no meaning for one-time
  quantities. Use a scalar factor, or move the lump share into a separate
  effect with a scalar factor.

If you need different cross-effect factors for the two domains, split into
separate effects.

### Transitive Chains

Contributions chain transitively. A PE → CO₂ → cost chain is modeled as:

### Validation

- **No self-references**: an effect cannot reference itself (\(\alpha_{k,k}\)).
- **No cycles**: \(k \to j \to \cdots \to k\) is rejected at build time to
  prevent singular systems.

### Pricing: `contribution_from` vs. objective weights

Two mechanisms can price one effect into another, and they mean different
things:

- **`contribution_from`** changes what the target effect *is*. Reported
  totals include the priced share, and `total_max` / `periodic_max` on the
  target bind against it. Use it when the price belongs in the accounting —
  e.g. an internal carbon price that should appear in reported cost.
- **Objective weights** (`optimize(..., objective={'cost': 1, 'co2': 250})`)
  change only the objective. Each effect's reported totals stay pure; the
  solver merely trades them off at the given rate. Use them for
  multi-criteria studies where the accounting must stay untouched.

Rule of thumb: if the number would appear in a financial report, use
`contribution_from`; if it is a study assumption, weight the objective.

## Total Aggregation

The total effect for each period \(p\) combines both domains:

\[
\Phi_{k,p} = \sum_{t \in \mathcal{T}} \Phi_{k,t,p}^{\text{temporal}} \cdot \mathrm{w}_t + \Phi_{k,p}^{\text{lump}} \quad \forall \, k \in \mathcal{K}, \; p \in \mathcal{P}
\]

Weights \(\mathrm{w}_t\) allow scaling timesteps (e.g., a representative week scaled to a year).
Single-period models drop the \(p\) index.

## Bounds

Two levels of bound granularity, all per-effect:

**Per-period bounds** (`periodic_max` / `periodic_min`) — each period independently:

\[
\underline{\Phi}_{k,p} \leq \Phi_{k,p} \leq \bar{\Phi}_{k,p} \quad \forall \, p
\]

**Aggregate bounds** (`total_max` / `total_min`) — weighted sum across all periods, where
\(\omega_{k,p}\) is `Effect.period_weights` (falling back to global `period_weights`, then 1):

\[
\underline{\Phi}_k \leq \sum_{p \in \mathcal{P}} \omega_{k,p} \cdot \Phi_{k,p} \leq \bar{\Phi}_k
\]

For physical quantities (e.g., total CO₂ across all years), `period_weights` should
typically encode the period duration so the aggregate is a true physical sum.

This is useful for emission caps or budget constraints.

Per-timestep effect bounds do not exist: nothing binds effects per timestep,
which is what allows the temporal domain to stay expression-only (temporal
closure — see `docs/design/model-data-tree.md` §2.5).

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\Phi_{k,t(,p)}^{\text{temporal}}\) | Per-timestep effect expression (folded into totals at build) | reconstructed in results |
| \(\Phi_{k(,p)}^{\text{lump}}\) | Lump effect variable (sizing + one-time costs) | `effect--lump[effect(, period)]` |
| \(\Phi_{k(,p)}\) | Total effect variable | `effect--total[effect(, period)]` |
| \(\mathrm{c}_{f,k,t}\) | Effect coefficient per flow-hour | [`Flow.effects_per_flow_hour`](../api/fluxopt/elements.md#fluxopt.elements.Flow.effects_per_flow_hour) |
| \(\alpha_{k,j,t}\) | Cross-effect contribution factor (time-varying) | [`Effect.contribution_from`](../api/fluxopt/elements.md#fluxopt.elements.Effect.contribution_from) (Variate) |
| \(\alpha_{k,j}\) | Cross-effect contribution factor (scalar) | [`Effect.contribution_from`](../api/fluxopt/elements.md#fluxopt.elements.Effect.contribution_from) (scalar) |
| \(P_{f,t}\) | Flow rate variable | `flow--rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(\mathrm{w}_t\) | Timestep weight | weights |
| \(\bar{\Phi}_k\) | Maximum aggregate (weighted sum across periods) | [`Effect.total_max`](../api/fluxopt/elements.md#fluxopt.elements.Effect.total_max) |
| \(\underline{\Phi}_k\) | Minimum aggregate (weighted sum across periods) | [`Effect.total_min`](../api/fluxopt/elements.md#fluxopt.elements.Effect.total_min) |
| \(\bar{\Phi}_{k,p}\) | Maximum per period (scalar or per-period values) | [`Effect.periodic_max`](../api/fluxopt/elements.md#fluxopt.elements.Effect.periodic_max) |
| \(\underline{\Phi}_{k,p}\) | Minimum per period (scalar or per-period values) | [`Effect.periodic_min`](../api/fluxopt/elements.md#fluxopt.elements.Effect.periodic_min) |
| \(\omega_{k,p}\) | Period weight (per-effect, falls back to global, then 1) | [`Effect.period_weights`](../api/fluxopt/elements.md#fluxopt.elements.Effect.period_weights) / global `period_weights` |

See [Notation](notation.md) for the full symbol table.

## Examples

### Direct effects

A system with two effects — cost (objective) and CO₂ (capped at 1000 kg):

A gas flow with both effect coefficients:

At timestep \(t\) with \(P_{\text{gas},t} = 5\) MW and \(\Delta t = 1\) h:

- \(\Phi_{\text{cost},t} = 30 \times 5 \times 1 = 150\) €
- \(\Phi_{\text{CO₂},t} = 0.2 \times 5 \times 1 = 1.0\) kg

### Carbon pricing via `contribution_from`

CO₂ priced at 50 €/t into the cost effect:

With \(\alpha_{\text{cost,co2}} = 50\), the per-timestep cost becomes:

\[
\Phi_{\text{cost},t} = \mathrm{c}_{\text{cost}} \cdot P_t \cdot \Delta t + 50 \cdot \Phi_{\text{co2},t}
\]

The CO₂ total itself is **not** affected — `contribution_from` is one-directional.
