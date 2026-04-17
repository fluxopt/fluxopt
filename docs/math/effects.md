# Effect Tracking & Bounding

## Overview

Effects represent quantities that are tracked across the optimization horizon (e.g.,
cost, CO₂ emissions, primary energy). One effect is designated as the objective to
minimize.

Effects are split into two **domains** based on how they vary over time and
how they are weighted in multi-period optimization:

| Domain | Dims | What goes here | Multi-period weighting |
|---|---|---|---|
| **Temporal** | `(effect, time)` | Flow costs, running costs, startup costs — anything that varies per timestep | Summed over time (× \(w_t\)), then weighted by `period_weights` |
| **Lump** | `(effect,)` | Sizing costs, fixed O&M, one-time CAPEX — anything not per-timestep | Weighted by \(\omega_{k,p}\) (defaults to global `period_weights`) |

All domains support cross-effect chains via `contribution_from`.

In multi-period mode, all variables gain an optional `period` dimension.
See [Objective](objective.md) for how the domains are weighted in the objective.

## Temporal Domain

Each effect accumulates contributions from all flows at each timestep:

\[
\Phi_{k,t}^{\text{temporal}} = \underbrace{\sum_{f \in \mathcal{F}} c_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t}_{\text{direct flow contributions}} + \underbrace{\sum_{j \in \mathcal{K}} \alpha_{k,j,t} \cdot \Phi_{j,t}^{\text{temporal}}}_{\text{cross-effect contributions}} \quad \forall \, k, t
\]

The coefficient \(c_{f,k,t}\) specifies how much of effect \(k\) is produced per
flow-hour of flow \(f\) (e.g., €/MWh for cost, kg/MWh for emissions).

The cross-effect factor \(\alpha_{k,j,t}\) can be time-varying or constant
(both via `contribution_from`).
Because \(\Phi_{k,t}^{\text{temporal}}\) is a **variable**, the solver resolves
multi-level chains (e.g., PE → CO₂ → cost) automatically.

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

The scalar factor \(\alpha_{k,j}\) from `contribution_from` applies to **both**
domains. Time-varying values in `contribution_from` apply to the temporal domain only;
the lump domain uses the scalar value.

### Validation

Self-references (\(\alpha_{k,k}\)) and circular dependencies
(\(k \to j \to \cdots \to k\)) are rejected at build time to prevent singular systems.

## Total Aggregation

The total effect combines both domains:

\[
\Phi_{k(,p)} = \sum_{t \in \mathcal{T}} \Phi_{k,t(,p)}^{\text{temporal}} \cdot w_t + \Phi_{k(,p)}^{\text{lump}} \quad \forall \, k \in \mathcal{K}
\]

Weights \(w_t\) allow scaling timesteps (e.g., a representative week scaled to a year).

## Total Bounds

Upper and lower bounds on the total effect over the entire horizon:

\[
\underline{\Phi}_k \leq \Phi_k \leq \bar{\Phi}_k
\]

This is useful for emission caps or budget constraints.

## Per-Hour Bounds

The per-hour bounds are **rates** (e.g., kg/h, €/h) that scale with the timestep
duration \(\Delta t_t\). This ensures the constraint is resolution-independent:

\[
\underline{\Phi}_{k,t}^{\text{per hour}} \cdot \Delta t_t \leq \Phi_{k,t}^{\text{temporal}} \leq \bar{\Phi}_{k,t}^{\text{per hour}} \cdot \Delta t_t \quad \forall \, k \in \mathcal{K}, \; t \in \mathcal{T}
\]

For example, `maximum_per_hour=100` (kg/h) with a 4-hour timestep allows up to
400 kg of emissions in that timestep.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\Phi_{k,t(,p)}^{\text{temporal}}\) | Per-timestep effect variable | `effect_temporal[effect, time(, period)]` |
| \(\Phi_{k(,p)}^{\text{lump}}\) | Lump effect variable (sizing + one-time costs) | `effect_lump[effect(, period)]` |
| \(\Phi_{k(,p)}\) | Total effect variable | `effect_total[effect(, period)]` |
| \(c_{f,k,t}\) | Effect coefficient per flow-hour | `Flow.effects_per_flow_hour` |
| \(\alpha_{k,j,t}\) | Cross-effect contribution factor (time-varying) | `Effect.contribution_from` (TimeSeries) |
| \(\alpha_{k,j}\) | Cross-effect contribution factor (scalar) | `Effect.contribution_from` (scalar) |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(w_t\) | Timestep weight | weights |
| \(\bar{\Phi}_k\) | Maximum total | `Effect.maximum` |
| \(\underline{\Phi}_k\) | Minimum total | `Effect.minimum` |
| \(\bar{\Phi}_{k,t}\) | Maximum per hour | `Effect.maximum_per_hour` |
| \(\underline{\Phi}_{k,t}\) | Minimum per hour | `Effect.minimum_per_hour` |

See [Notation](notation.md) for the full symbol table.

## Examples

### Direct effects

A system with two effects — cost (objective) and CO₂ (capped at 1000 kg):

```python
effects = [
    Effect("cost", unit="€"),
    Effect("CO2", unit="kg", maximum=1000),
]
```

A gas flow with both effect coefficients:

```python
gas_flow = Flow("gas", bus="gas_bus", effects_per_flow_hour={"cost": 30, "CO2": 0.2})
```

At timestep \(t\) with \(P_{\text{gas},t} = 5\) MW and \(\Delta t = 1\) h:

- \(\Phi_{\text{cost},t} = 30 \times 5 \times 1 = 150\) €
- \(\Phi_{\text{CO₂},t} = 0.2 \times 5 \times 1 = 1.0\) kg

### Carbon pricing via `contribution_from`

CO₂ priced at 50 €/t into the cost effect:

```python
effects = [
    Effect("cost", contribution_from={"co2": 50}),
    Effect("co2", unit="kg"),
]
```

With \(\alpha_{\text{cost,co2}} = 50\), the per-timestep cost becomes:

\[
\Phi_{\text{cost},t} = c_{\text{cost}} \cdot P_t \cdot \Delta t + 50 \cdot \Phi_{\text{co2},t}
\]

The CO₂ total itself is **not** affected — `contribution_from` is one-directional.
