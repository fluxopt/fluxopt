# Objective Function

## Formulation

The model minimizes the total value of the designated objective effect \(k^*\)
(specified via `optimize(objective_effects='cost')`).

### Single-period

Without periods, the objective is simply:

\[
\min \; \Phi_{k^*}
\]

### Multi-period

With periods \(p \in \mathcal{P}\), the objective weights each period's total effect:

\[
\min \; \sum_{p \in \mathcal{P}} \omega_{k^*,p} \cdot \left(\sum_t \Phi_{k^*,t,p}^{\text{temporal}} \cdot \mathrm{w}_t + \Phi_{k^*,p}^{\text{lump}}\right)
\]

- \(\omega_{k,p}\) defaults to the global `period_weights` (inferred from
  gaps between period labels, or explicit). Override per effect via `Effect.period_weights`.

This allows different weighting strategies per effect (e.g., NPV discounting for costs,
flat weights for emissions).

### Total effect

The total effect per period combines both domains:

\[
\Phi_{k(,p)} = \sum_{t \in \mathcal{T}} \Phi_{k,t(,p)}^{\text{temporal}} \cdot \mathrm{w}_t + \Phi_{k(,p)}^{\text{lump}}
\]

The **temporal** domain accumulates flow contributions, running costs,
startup costs, and cross-effect contributions per timestep:

\[
\Phi_{k,t}^{\text{temporal}} = \underbrace{\sum_{f} \mathrm{c}_{f,k,t} \cdot P_{f,t} \cdot \Delta t_t}_{\text{flow}} + \underbrace{\sum_{f} \mathrm{r}_{f,k,t} \cdot \sigma_{f,t} \cdot \Delta t_t}_{\text{running}} + \underbrace{\sum_{f} \mathrm{u}_{f,k,t} \cdot \tau^+_{f,t}}_{\text{startup}} + \underbrace{\sum_{j} \alpha_{k,j,t} \cdot \Phi_{j,t}^{\text{temporal}}}_{\text{cross-effect}}
\]

The **lump** domain accumulates sizing costs, fixed costs, one-time costs, and cross-effect contributions:

\[
\Phi_k^{\text{lump}} = \underbrace{\sum_{f} \gamma_{f,k} \cdot S_f + \sum_{f} \phi_{f,k} \cdot y_f + \sum_{s} \gamma_{s,k} \cdot S_s + \sum_{s} \phi_{s,k} \cdot y_s}_{\text{direct sizing costs}} + \underbrace{\sum_{j} \alpha_{k,j} \cdot \Phi_j^{\text{lump}}}_{\text{cross-effect}}
\]

See [Sizing](sizing.md), [Status](status.md), and [Effects](effects.md) for
full formulations of each term.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(k^*\) | Objective effect | `optimize(objective_effects='cost')` |
| \(\mathrm{c}_{f,k,t}\) | Effect coefficient per flow-hour | [`Flow.effects_per_flow_hour`](../api/fluxopt/elements.md#fluxopt.elements.Flow(effects_per_flow_hour)) |
| \(P_{f,t}\) | Flow rate variable | `flow--rate[flow, time]` |
| \(\Delta t_t\) | Timestep duration | dt |
| \(\mathrm{w}_t\) | Timestep weight | weights |
| \(\omega_{k,p}\) | Period weight | [`Effect.period_weights`](../api/fluxopt/elements.md#fluxopt.elements.Effect(period_weights)) (fallback: [`optimize(period_weights=...)`](../api/fluxopt/index.md#fluxopt.optimize(period_weights))) |
| \(\Phi_{k,t(,p)}^{\text{temporal}}\) | Temporal (per-timestep) effect expression | folded into totals at build |
| \(\Phi_{k(,p)}^{\text{lump}}\) | Lump effect variable (sizing + one-time costs) | `effect--lump[effect(, period)]` |
| \(\Phi_{k(,p)}\) | Total effect variable | `effect--total[effect(, period)]` |

See [Notation](notation.md) for the full symbol table.

## Examples

### Single-period

Consider a gas boiler over 3 timesteps (\(\Delta t = 1\,\text{h}\), \(w = 1\)):

| \(t\) | \(P_{\text{gas},t}\) (MW) | \(\mathrm{c}_{\text{gas,cost}}\) (€/MWh) | \(\Phi_{\text{cost},t}^{\text{temporal}}\) (€) |
|---|---|---|---|
| 1 | 2.0 | 30 | \(30 \times 2.0 \times 1 = 60\) |
| 2 | 3.0 | 30 | \(30 \times 3.0 \times 1 = 90\) |
| 3 | 1.5 | 30 | \(30 \times 1.5 \times 1 = 45\) |

Total cost: \(\Phi_{\text{cost}} = \sum_t \Phi_{\text{cost},t}^{\text{temporal}} = 60 + 90 + 45 = 195\,\text{€}\)

### Multi-period

Same system with `periods=[2020, 2025]`, `period_weights=[5, 5]`.

Each period has the same 3 timesteps with per-period cost = 30 €. The objective becomes:

\[
\sum_p \omega_p \cdot \Phi_{\text{cost},p} = 5 \times 30 + 5 \times 30 = 300\,\text{€}
\]

The optimizer finds the \(P_{f,t(,p)}\) values that minimize the (period-weighted)
\(\Phi_{k^*}\) subject to all constraints.
