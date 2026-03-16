# Effect Tracking & Bounding

## Overview

Effects represent quantities that are tracked across the optimization horizon (e.g.,
cost, CO₂ emissions, primary energy). One effect is designated as the objective to
minimize.

Effects are split into three **domains** based on how they vary over time and
how they are weighted in multi-period optimization:

| Domain | Dims | What goes here | Multi-period weighting |
|---|---|---|---|
| **Temporal** | `(effect, time)` | Flow costs, running costs, startup costs — anything that varies per timestep | Summed over time (× \(w_t\)), then weighted like periodic |
| **Periodic** | `(effect,)` | Recurring costs that repeat each period — sizing costs, fixed annual O&M | Weighted by \(\omega^{\text{periodic}}_{k,p}\) (defaults to global `period_weights`) |
| **Once** | `(effect,)` | One-time costs at a point in time — CAPEX, decommissioning | Weighted by \(\omega^{\text{once}}_{k,p}\) (defaults to 1, no scaling) |

The key distinction: **periodic** costs are assumed to recur across the gap between
periods (e.g., annual O&M for 5 years), while **once** costs happen at a specific point
(e.g., an investment decision in 2025). This matters because their period weights
differ — recurring costs scale with duration, one-time costs typically don't
(or use discount factors instead).

All domains support cross-effect chains via `cross_temporal`, `cross_periodic`,
and `cross_once`.

In multi-period mode, all variables gain an optional `period` dimension.
See [Objective](objective.md) for how the domains are weighted in the objective.

## Temporal Domain

Each effect accumulates contributions from all flows at each timestep:

\[
\Phi_{k,t}^{\text{temporal}} = \underbrace{\sum_{f \in \mathcal{F}} c_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t}_{\text{direct flow contributions}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha_{k,j,t} \cdot \Phi_{j,t}^{\text{temporal}}}_{\text{cross-effect contributions}} \quad \forall \, k, t
\]

The coefficient \(c_{f,k,t}\) specifies how much of effect \(k\) is produced per
flow-hour of flow \(f\) (e.g., €/MWh for cost, kg/MWh for emissions).

The cross-effect factor \(\alpha_{k,j,t}\) is set via `cross_temporal` and
can be time-varying or constant.
Because \(\Phi_{k,t}^{\text{temporal}}\) is a **variable**, the solver resolves
multi-level chains (e.g., PE → CO₂ → cost) automatically.

## Periodic Domain

Sizing costs and fixed costs (not time-varying) are accumulated per effect:

\[
\Phi_k^{\text{periodic}} = \underbrace{\Phi_k^{\text{invest,direct}}}_{\text{direct sizing costs}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha_{k,j} \cdot \Phi_j^{\text{periodic}}}_{\text{cross-effect contributions}} \quad \forall \, k
\]

where the direct investment term is:

\[
\Phi_k^{\text{invest,direct}} = \sum_{f} \gamma_{f,k} \cdot S_f + \sum_{f} \phi_{f,k} \cdot y_f + \sum_{s} \gamma_{s,k} \cdot S_s + \sum_{s} \phi_{s,k} \cdot y_s
\]

Because \(\Phi_k^{\text{periodic}}\) is a **variable** (not an expression), the
solver resolves multi-level chains correctly: if PE has sizing costs
and CO₂ depends on PE and cost depends on CO₂, the chain propagates through the
periodic domain just as it does through the temporal domain.

## Cross-Effect Contributions

An effect can include a weighted fraction of another effect's value. Each domain
has its own cross-effect field — `cross_temporal`, `cross_periodic`, and
`cross_once` — enabling patterns like carbon pricing (CO₂ → cost)
or transitive chains (PE → CO₂ → cost).

The factor \(\alpha_{k,j}\) from `cross_periodic` applies to the periodic domain,
\(\alpha_{k,j,t}\) from `cross_temporal` applies to the temporal domain (and can
be time-varying), and \(\alpha_{k,j}\) from `cross_once` applies to the once domain.

### Validation

Self-references (\(\alpha_{k,k}\)) and circular dependencies
(\(k \to j \to \cdots \to k\)) are rejected at build time to prevent singular systems.

## Once Domain

One-time costs that should not be scaled by period weights (e.g., investment CAPEX,
decommissioning costs):

\[
\Phi_k^{\text{once}} = \underbrace{\Phi_k^{\text{once,direct}}}_{\text{direct one-time costs}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha^{\text{once}}_{k,j} \cdot \Phi_j^{\text{once}}}_{\text{cross-effect contributions}} \quad \forall \, k
\]

The cross-effect factor \(\alpha^{\text{once}}_{k,j}\) is set via `cross_once`.

## Total Aggregation

The total effect combines all three domains:

\[
\Phi_{k(,p)} = \sum_{t \in \mathcal{T}} \Phi_{k,t(,p)}^{\text{temporal}} \cdot w_t + \Phi_{k(,p)}^{\text{periodic}} + \Phi_{k(,p)}^{\text{once}} \quad \forall \, k \in \mathcal{K}
\]

Weights \(w_t\) allow scaling timesteps (e.g., a representative week scaled to a year).

## Total Bounds

Upper and lower bounds on the total effect over the entire horizon:

\[
\underline{\Phi}_k \leq \Phi_k \leq \bar{\Phi}_k
\]

This is useful for emission caps or budget constraints.

## Per-Timestep Bounds

Bounds on the effect value at each timestep:

\[
\underline{\Phi}_{k,t} \leq \Phi_{k,t} \leq \bar{\Phi}_{k,t} \quad \forall \, t \in \mathcal{T}
\]

This enforces per-hour limits (e.g., maximum hourly emissions).

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\Phi_{k,t(,p)}^{\text{temporal}}\) | Per-timestep effect variable | `effect_temporal[effect, time(, period)]` |
| \(\Phi_{k(,p)}^{\text{periodic}}\) | Periodic effect variable (recurring costs) | `effect_periodic[effect(, period)]` |
| \(\Phi_{k(,p)}^{\text{once}}\) | One-time effect variable | `effect_once[effect(, period)]` |
| \(\Phi_{k(,p)}\) | Total effect variable | `effect_total[effect(, period)]` |
| \(c_{f,k,t}\) | Effect coefficient per flow-hour | `Flow.effects_per_flow_hour` |
| \(\alpha_{k,j,t}\) | Cross-effect factor (temporal, possibly time-varying) | `Effect.cross_temporal` |
| \(\alpha_{k,j}\) | Cross-effect factor (periodic) | `Effect.cross_periodic` |
| \(\alpha^{\text{once}}_{k,j}\) | Cross-effect factor (once) | `Effect.cross_once` |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(w_t\) | Timestep weight | weights |
| \(\bar{\Phi}_k\) | Maximum total | `Effect.maximum_total` |
| \(\underline{\Phi}_k\) | Minimum total | `Effect.minimum_total` |
| \(\bar{\Phi}_{k,t}\) | Maximum per hour | `Effect.maximum_per_hour` |
| \(\underline{\Phi}_{k,t}\) | Minimum per hour | `Effect.minimum_per_hour` |

See [Notation](notation.md) for the full symbol table.

## Examples

### Direct effects

A system with two effects — cost (objective) and CO₂ (capped at 1000 kg):

```python
effects = [
    Effect("cost", unit="€", is_objective=True),
    Effect("CO2", unit="kg", maximum_total=1000),
]
```

A gas flow with both effect coefficients:

```python
gas_flow = Flow("gas", bus="gas_bus", effects_per_flow_hour={"cost": 30, "CO2": 0.2})
```

At timestep \(t\) with \(P_{\text{gas},t} = 5\) MW and \(\Delta t = 1\) h:

- \(\Phi_{\text{cost},t} = 30 \times 5 \times 1 = 150\) €
- \(\Phi_{\text{CO₂},t} = 0.2 \times 5 \times 1 = 1.0\) kg

### Carbon pricing via cross-effects

CO₂ priced at 50 €/t into the cost effect (temporal domain):

```python
effects = [
    Effect("cost", is_objective=True, cross_temporal={"co2": 50}),
    Effect("co2", unit="kg"),
]
```

With \(\alpha_{\text{cost,co2}} = 50\), the per-timestep cost becomes:

\[
\Phi_{\text{cost},t} = c_{\text{cost}} \cdot P_t \cdot \Delta t + 50 \cdot \Phi_{\text{co2},t}
\]

The CO₂ total itself is **not** affected — cross-effects are one-directional.
