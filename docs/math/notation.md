# Notation

This page defines the canonical symbols used throughout the mathematical formulation.
Each symbol maps to a specific field or variable in the code.

## Sets & Indices

| Symbol | Description | Code |
|---|---|---|
| \(t \in \mathcal{T}\) | Timesteps | `time` dimension |
| \(p \in \mathcal{P}\) | Periods (multi-period only) | `period` dimension |
| \(f \in \mathcal{F}\) | Flows | `flow` dimension |
| \(b \in \mathcal{B}\) | Buses | `bus` dimension |
| \(s \in \mathcal{S}\) | Storages | `storage` dimension |
| \(k \in \mathcal{K}\) | Effects (cost, CO₂, …) | `effect` dimension |
| \(j \in \mathcal{K}\) | Source effect (cross-effect) | `source_effect` dimension |

## Variables

| Symbol | Code | Domain | Unit | Description |
|---|---|---|---|---|
| \(P_{f,t(,p)}\) | `flow--rate[flow, time(, period)]` | \(\geq 0\) | MW | Flow rate |
| \(E_{s,t(,p)}\) | `storage--level[storage, time(, period)]` | \(\geq 0\) | MWh | Stored energy |
| \(\Phi_{k,t(,p)}^{\text{temporal}}\) | `effect--temporal[effect, time(, period)]` | \(\mathbb{R}\) | varies | Temporal (per-timestep) effect |
| \(\Phi_{k(,p)}^{\text{lump}}\) | `effect--lump[effect(, period)]` | \(\mathbb{R}\) | varies | Lump (sizing + one-time) effect |
| \(\Phi_{k(,p)}\) | `effect--total[effect(, period)]` | \(\mathbb{R}\) | varies | Total effect per period |
| \(S_{f(,p)}\) | `flow--size[flow(, period)]` | \(\geq 0\) | MW | Invested flow capacity |
| \(y_{f(,p)}\) | `flow--size_indicator[flow(, period)]` | \(\{0, 1\}\) | — | Binary invest indicator (flow) |
| \(S_{s(,p)}\) | `storage--capacity[storage(, period)]` | \(\geq 0\) | MWh | Invested storage capacity |
| \(y_{s(,p)}\) | `storage--size_indicator[storage(, period)]` | \(\{0, 1\}\) | — | Binary invest indicator (storage) |
| \(\sigma_{f,t(,p)}\) | `flow--on[flow, time(, period)]` | \(\{0, 1\}\) | — | On/off indicator |
| \(\tau^+_{f,t(,p)}\) | `flow--startup[flow, time(, period)]` | \(\{0, 1\}\) | — | Startup event indicator |
| \(\tau^-_{f,t(,p)}\) | `flow--shutdown[flow, time(, period)]` | \(\{0, 1\}\) | — | Shutdown event indicator |
| \(\mathrm{D}^{\text{up}}_{f,t(,p)}\) | `uptime[flow, time(, period)]` | \(\geq 0\) | h | Consecutive uptime |
| \(\mathrm{D}^{\text{down}}_{f,t(,p)}\) | `downtime[flow, time(, period)]` | \(\geq 0\) | h | Consecutive downtime |

## Indexing Convention

**Variables** (the table above) are shown with **all** dimensions they
live on. There's no broadcasting for decision variables — every
\((\text{flow}, \text{time}, \ldots)\) cell is its own variable.

**Optional dims** — currently just period — exist in some models and
not others. We mark them in parentheses, e.g. \(P_{f,t(,p)}\): the
period subscript is *present* in multi-period models, *absent* in
single-period ones. To keep formulas readable, constraint derivations
that aren't specifically about multi-period dynamics drop the
parenthetical and treat \(p\) as implicit context — the constraint
applies pointwise across whichever optional dims happen to be in the
model.

**Parameters** are different. Symbol subscripts show only the indices
the formulation **structurally requires** — the dims a constraint
iterates over and that change its meaning. Most parameters also
**broadcast over additional dims** in the API; we don't decorate the
symbols with every dim they *could* take, because that turns formulas
into pyramids of subscripts.

The broadcast hierarchy:

- **Period (\(p\))** — almost every parameter accepts this: even
  "scalar" things like \(\bar{\mathrm{P}}_f\), \(\mathrm{D}^{\text{up,min}}\), or
  \(\mathrm{S}^-\) become \(\bar{\mathrm{P}}_{f,p}\) etc. in multi-period models.
  Variables already reflect this in the table above (e.g.
  \(P_{f,t(,p)}\)).
- **Time (\(t\))** — only fields typed `Variate` accept this: bounds
  \(\underline{\mathrm{p}}_{f,t}, \bar{\mathrm{p}}_{f,t}\), profiles \(\pi_{f,t}\),
  efficiencies \(\eta^c_s, \eta^d_s\), losses \(\delta_s\), conversion
  coefficients \(\mathrm{a}_{f,i}\), effect / running / startup costs
  \(\mathrm{c}_{f,k,t}, \mathrm{r}_{f,k,t}, \mathrm{u}_{f,k,t}\), cross-effects \(\alpha_{k,j,t}\).
- **Build period (\(p_b\))** — at-build investment coefficients only:
  \(\gamma^{\text{build}}_{f,k}, \phi^{\text{build}}_{f,k}\), …

When a formula shows a \(t\) subscript on a parameter, it's because
time variation is structurally meaningful for that constraint (e.g. a
fixed profile is meaningless without time). Parameters without a \(t\)
subscript may still be time-varying when the API field accepts it —
the constraint just reads it pointwise. The same applies for \(p\):
omit unless we're discussing multi-period dynamics specifically.

## Parameters

| Symbol | Code | Domain | Unit | Description |
|---|---|---|---|---|
| \(\bar{\mathrm{P}}_f\) | [`Flow.size`](../api/fluxopt/elements.md#fluxopt.elements.Flow.size) | \(\geq 0\) or \(\infty\) | MW | Nominal capacity |
| \(\underline{\mathrm{p}}_{f,t}\) | [`Flow.relative_rate_min`](../api/fluxopt/elements.md#fluxopt.elements.Flow.relative_rate_min) | \([0, 1]\) | — | Relative lower bound |
| \(\bar{\mathrm{p}}_{f,t}\) | [`Flow.relative_rate_max`](../api/fluxopt/elements.md#fluxopt.elements.Flow.relative_rate_max) | \([0, 1]\) | — | Relative upper bound |
| \(\pi_{f,t}\) | [`Flow.fixed_relative_profile`](../api/fluxopt/elements.md#fluxopt.elements.Flow.fixed_relative_profile) | \([0, 1]\) | — | Fixed profile |
| \(\mathrm{c}_{f,k,t}\) | [`Flow.effects_per_flow_hour`](../api/fluxopt/elements.md#fluxopt.elements.Flow.effects_per_flow_hour) | \(\mathbb{R}\) | varies | Effect coefficient per flow-hour |
| \(\bar{\mathrm{E}}_s\) | [`Storage.capacity`](../api/fluxopt/elements.md#fluxopt.elements.Storage.capacity) | \(\geq 0\) | MWh | Storage capacity |
| \(\eta^{\text{c}}_s\) | [`Storage.eta_charge`](../api/fluxopt/elements.md#fluxopt.elements.Storage.eta_charge) | \((0, 1]\) | — | Charging efficiency |
| \(\eta^{\text{d}}_s\) | [`Storage.eta_discharge`](../api/fluxopt/elements.md#fluxopt.elements.Storage.eta_discharge) | \((0, 1]\) | — | Discharging efficiency |
| \(\delta_s\) | [`Storage.relative_loss_per_hour`](../api/fluxopt/elements.md#fluxopt.elements.Storage.relative_loss_per_hour) | \([0, 1]\) | 1/h | Self-discharge rate |
| \(\underline{\mathrm{e}}_s\) | [`Storage.relative_level_min`](../api/fluxopt/elements.md#fluxopt.elements.Storage.relative_level_min) | \([0, 1]\) | — | Relative min SOC |
| \(\bar{\mathrm{e}}_s\) | [`Storage.relative_level_max`](../api/fluxopt/elements.md#fluxopt.elements.Storage.relative_level_max) | \([0, 1]\) | — | Relative max SOC |
| \(\mathrm{a}_{f,i}\) | [`Converter.conversion_factors`](../api/fluxopt/components.md#fluxopt.components.Converter.conversion_factors) | \(\mathbb{R}\) | — | Conversion coefficient (per flow, per equation) |
| \(\alpha_{k,j}\) | [`Effect.contribution_from`](../api/fluxopt/elements.md#fluxopt.elements.Effect.contribution_from) | \(\mathbb{R}\) | varies | Cross-effect factor (scalar) |
| \(\alpha_{k,j,t}\) | [`Effect.contribution_from`](../api/fluxopt/elements.md#fluxopt.elements.Effect.contribution_from) (Variate) | \(\mathbb{R}\) | varies | Cross-effect factor (time-varying; lump uses time-mean) |
| \(\bar{\Phi}_k\) | [`Effect.total_max`](../api/fluxopt/elements.md#fluxopt.elements.Effect.total_max) | \(\mathbb{R}\) | varies | Maximum aggregate (weighted sum across periods) |
| \(\underline{\Phi}_k\) | [`Effect.total_min`](../api/fluxopt/elements.md#fluxopt.elements.Effect.total_min) | \(\mathbb{R}\) | varies | Minimum aggregate (weighted sum across periods) |
| \(\bar{\Phi}_k^{\text{per period}}\) | [`Effect.periodic_max`](../api/fluxopt/elements.md#fluxopt.elements.Effect.periodic_max) | \(\mathbb{R}\) | varies | Maximum per period |
| \(\underline{\Phi}_k^{\text{per period}}\) | [`Effect.periodic_min`](../api/fluxopt/elements.md#fluxopt.elements.Effect.periodic_min) | \(\mathbb{R}\) | varies | Minimum per period |
| \(\bar{\Phi}_{k,t}^{\text{per hour}}\) | [`Effect.rate_max`](../api/fluxopt/elements.md#fluxopt.elements.Effect.rate_max) | \(\mathbb{R}\) | varies/h | Maximum per hour (rate, scaled by \(\Delta t_t\)) |
| \(\underline{\Phi}_{k,t}^{\text{per hour}}\) | [`Effect.rate_min`](../api/fluxopt/elements.md#fluxopt.elements.Effect.rate_min) | \(\mathbb{R}\) | varies/h | Minimum per hour (rate, scaled by \(\Delta t_t\)) |
| \(\mathrm{S}^-\) | [`Sizing.size_min`](../api/fluxopt/elements.md#fluxopt.elements.Sizing.size_min) | \(\geq 0\) | MW or MWh | Minimum invested size (flow or storage) |
| \(\mathrm{S}^+\) | [`Sizing.size_max`](../api/fluxopt/elements.md#fluxopt.elements.Sizing.size_max) | \(\geq 0\) | MW or MWh | Maximum invested size (flow or storage) |
| \(\gamma_{f,k}\), \(\gamma_{s,k}\) | [`Sizing.effects_per_size`](../api/fluxopt/elements.md#fluxopt.elements.Sizing.effects_per_size) | \(\mathbb{R}\) | varies | Per-size investment cost (flow or storage; one-time, sized) |
| \(\phi_{f,k}\), \(\phi_{s,k}\) | [`Sizing.effects_fixed`](../api/fluxopt/elements.md#fluxopt.elements.Sizing.effects_fixed) | \(\mathbb{R}\) | varies | Fixed investment cost (flow or storage; one-time, sized) |
| \(\gamma^{\text{build}}_{f,k}\) | [`Investment.effects_per_size_at_build`](../api/fluxopt/elements.md#fluxopt.elements.Investment.effects_per_size_at_build) | \(\mathbb{R}\) | varies | Per-size CAPEX charged in the build period |
| \(\phi^{\text{build}}_{f,k}\) | [`Investment.effects_fixed_at_build`](../api/fluxopt/elements.md#fluxopt.elements.Investment.effects_fixed_at_build) | \(\mathbb{R}\) | varies | Fixed CAPEX charged in the build period |
| \(\gamma^{\text{rec}}_{f,k}\) | [`Investment.effects_per_size_recurring`](../api/fluxopt/elements.md#fluxopt.elements.Investment.effects_per_size_recurring) | \(\mathbb{R}\) | varies | Recurring per-size cost (each active period) |
| \(\phi^{\text{rec}}_{f,k}\) | [`Investment.effects_fixed_recurring`](../api/fluxopt/elements.md#fluxopt.elements.Investment.effects_fixed_recurring) | \(\mathbb{R}\) | varies | Recurring fixed cost (each active period) |
| \(\mathrm{D}^{\text{up,min}}\) | [`Status.uptime_min`](../api/fluxopt/elements.md#fluxopt.elements.Status.uptime_min) | \(\geq 0\) | h | Minimum consecutive uptime |
| \(\mathrm{D}^{\text{up,max}}\) | [`Status.uptime_max`](../api/fluxopt/elements.md#fluxopt.elements.Status.uptime_max) | \(\geq 0\) | h | Maximum consecutive uptime |
| \(\mathrm{D}^{\text{down,min}}\) | [`Status.downtime_min`](../api/fluxopt/elements.md#fluxopt.elements.Status.downtime_min) | \(\geq 0\) | h | Minimum consecutive downtime |
| \(\mathrm{D}^{\text{down,max}}\) | [`Status.downtime_max`](../api/fluxopt/elements.md#fluxopt.elements.Status.downtime_max) | \(\geq 0\) | h | Maximum consecutive downtime |
| \(\mathrm{r}_{f,k,t}\) | [`Status.effects_per_running_hour`](../api/fluxopt/elements.md#fluxopt.elements.Status.effects_per_running_hour) | \(\mathbb{R}\) | varies | Running cost coefficient |
| \(\mathrm{u}_{f,k,t}\) | [`Status.effects_per_startup`](../api/fluxopt/elements.md#fluxopt.elements.Status.effects_per_startup) | \(\mathbb{R}\) | varies | Startup cost coefficient |
| \(\mathrm{w}_t\) | weights | \(> 0\) | — | Timestep weight |
| \(\Delta t_t\) | dt | \(> 0\) | h | Timestep duration |
| \(\omega_p\) | [`optimize(period_weights=...)`](../api/fluxopt/index.md#fluxopt.optimize(period_weights)) | \(> 0\) | — | Global period weight (multi-period only) |
| \(\omega_{k,p}\) | [`Effect.period_weights`](../api/fluxopt/elements.md#fluxopt.elements.Effect.period_weights) | \(> 0\) | — | Per-effect period weight |

## Naming Conventions

**Italic letters denote decision variables; upright (`\mathrm{}`) letters
denote parameters** — following ISO 80000-2. So \(P_{f,t}\) (italic) is the
flow rate variable, while \(\bar{\mathrm{P}}_f\) (upright with overbar) is the
parameter that bounds it.

| Convention | Meaning | Example |
|---|---|---|
| Italic (default math mode) | Decision variables | \(P\) (flow rate), \(E\) (stored energy), \(S\) (size) |
| Upright (`\mathrm{}`) | Parameters | \(\mathrm{c}_{f,k,t}\) (cost coefficient), \(\bar{\mathrm{P}}_f\) (capacity), \(\mathrm{a}_{f,i}\) (conversion coefficient) |
| Overbar / underbar | Bound (paired with the variable's letter) | \(\bar{\mathrm{P}}\) (upper bound on \(P\)), \(\underline{\mathrm{P}}\) (lower bound) |
| Lowercase variant | Relative bound (fraction of size/capacity) | \(\bar{\mathrm{p}}_{f,t}\), \(\underline{\mathrm{p}}_{f,t}\) (fraction of \(\bar{\mathrm{P}}_f\)); \(\bar{\mathrm{e}}_s\), \(\underline{\mathrm{e}}_s\) (fraction of \(\bar{\mathrm{E}}_s\)) |
| Greek | Parameters with established physical meaning | \(\eta\) (efficiency), \(\delta\) (loss), \(\pi\) (profile), \(\gamma\) (per-size cost) |
| Subscripts | Indexing | \(f\) (flow), \(t\) (time), \(s\) (storage), \(b\) (bus), \(k\) (effect), \(j\) (source effect) |
| Superscripts | Qualification | \(\eta^{\text{c}}\) (charge), \(\eta^{\text{d}}\) (discharge) |

Greek letters render with their conventional shape regardless of `\mathrm{}`,
so we use them as-is for parameters without further wrapping.
