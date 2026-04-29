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
| \(S_{f(,p)}\) | `flow--size[flow(, period)]` | \(\geq 0\) | MW | Flow capacity |
| \(y_{f(,p)}\) | `flow--size_indicator[flow(, period)]` | \(\{0, 1\}\) | — | Binary invest indicator (flow) |
| \(S_{s(,p)}\) | `storage--capacity[storage(, period)]` | \(\geq 0\) | MWh | Storage capacity |
| \(y_{s(,p)}\) | `storage--size_indicator[storage(, period)]` | \(\{0, 1\}\) | — | Binary invest indicator (storage) |
| \(\sigma_{f,t(,p)}\) | `flow--on[flow, time(, period)]` | \(\{0, 1\}\) | — | On/off indicator |
| \(\tau^+_{f,t(,p)}\) | `flow--startup[flow, time(, period)]` | \(\{0, 1\}\) | — | Startup event indicator |
| \(\tau^-_{f,t(,p)}\) | `flow--shutdown[flow, time(, period)]` | \(\{0, 1\}\) | — | Shutdown event indicator |
| \(D^{\text{up}}_{f,t(,p)}\) | `uptime[flow, time(, period)]` | \(\geq 0\) | h | Consecutive uptime |
| \(D^{\text{down}}_{f,t(,p)}\) | `downtime[flow, time(, period)]` | \(\geq 0\) | h | Consecutive downtime |

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
  "scalar" things like \(\bar{P}_f\), \(D^{\text{up,min}}\), or
  \(S^-\) become \(\bar{P}_{f,p}\) etc. in multi-period models.
  Variables already reflect this in the table above (e.g.
  \(P_{f,t(,p)}\)).
- **Time (\(t\))** — only fields typed `TimeSeries` accept this: bounds
  \(\underline{p}_{f,t}, \bar{p}_{f,t}\), profiles \(\pi_{f,t}\),
  efficiencies \(\eta^c_s, \eta^d_s\), losses \(\delta_s\), conversion
  coefficients \(a_{f,i}\), effect / running / startup costs
  \(c_{f,k,t}, r_{f,k,t}, u_{f,k,t}\), cross-effects \(\alpha_{k,j,t}\).
- **Build period (\(p_b\))** — investment-domain coefficients only:
  \(\gamma^{\text{build}}_{f,k}, \phi^{\text{rec}}_{f,k}\), …

When a formula shows a \(t\) subscript on a parameter, it's because
time variation is structurally meaningful for that constraint (e.g. a
fixed profile is meaningless without time). Parameters without a \(t\)
subscript may still be time-varying when the API field accepts it —
the constraint just reads it pointwise. The same applies for \(p\):
omit unless we're discussing multi-period dynamics specifically.

## Parameters

| Symbol | Code | Domain | Unit | Description |
|---|---|---|---|---|
| \(\bar{P}_f\) | `Flow.size` | \(\geq 0\) or \(\infty\) | MW | Nominal capacity |
| \(\underline{p}_{f,t}\) | `Flow.relative_minimum` | \([0, 1]\) | — | Relative lower bound |
| \(\bar{p}_{f,t}\) | `Flow.relative_maximum` | \([0, 1]\) | — | Relative upper bound |
| \(\pi_{f,t}\) | `Flow.fixed_relative_profile` | \([0, 1]\) | — | Fixed profile |
| \(c_{f,k,t}\) | `Flow.effects_per_flow_hour` | \(\mathbb{R}\) | varies | Effect coefficient per flow-hour |
| \(\bar{E}_s\) | `Storage.capacity` | \(\geq 0\) | MWh | Storage capacity |
| \(\eta^{\text{c}}_s\) | `Storage.eta_charge` | \((0, 1]\) | — | Charging efficiency |
| \(\eta^{\text{d}}_s\) | `Storage.eta_discharge` | \((0, 1]\) | — | Discharging efficiency |
| \(\delta_s\) | `Storage.relative_loss_per_hour` | \([0, 1]\) | 1/h | Self-discharge rate |
| \(\underline{e}_s\) | `Storage.relative_minimum_level` | \([0, 1]\) | — | Relative min SOC |
| \(\bar{e}_s\) | `Storage.relative_maximum_level` | \([0, 1]\) | — | Relative max SOC |
| \(a_{f,i}\) | `Converter.conversion_factors` | \(\mathbb{R}\) | — | Conversion coefficient (per flow, per equation) |
| \(\alpha_{k,j}\) | `Effect.contribution_from` | \(\mathbb{R}\) | varies | Cross-effect factor (scalar) |
| \(\alpha_{k,j,t}\) | `Effect.contribution_from` (TimeSeries) | \(\mathbb{R}\) | varies | Cross-effect factor (time-varying; lump uses time-mean) |
| \(\bar{\Phi}_k\) | `Effect.maximum` | \(\mathbb{R}\) | varies | Maximum aggregate (weighted sum across periods) |
| \(\underline{\Phi}_k\) | `Effect.minimum` | \(\mathbb{R}\) | varies | Minimum aggregate (weighted sum across periods) |
| \(\bar{\Phi}_k^{\text{per period}}\) | `Effect.maximum_per_period` | \(\mathbb{R}\) | varies | Maximum per period |
| \(\underline{\Phi}_k^{\text{per period}}\) | `Effect.minimum_per_period` | \(\mathbb{R}\) | varies | Minimum per period |
| \(\bar{\Phi}_{k,t}^{\text{per hour}}\) | `Effect.maximum_per_hour` | \(\mathbb{R}\) | varies/h | Maximum per hour (rate, scaled by \(\Delta t_t\)) |
| \(\underline{\Phi}_{k,t}^{\text{per hour}}\) | `Effect.minimum_per_hour` | \(\mathbb{R}\) | varies/h | Minimum per hour (rate, scaled by \(\Delta t_t\)) |
| \(S^-\) | `Sizing.min_size` | \(\geq 0\) | MW or MWh | Minimum invested size (flow or storage) |
| \(S^+\) | `Sizing.max_size` | \(\geq 0\) | MW or MWh | Maximum invested size (flow or storage) |
| \(\gamma_{f,k}\) | `Sizing.effects_per_size` | \(\mathbb{R}\) | varies | Per-size investment cost (one-time, sized) |
| \(\phi_{f,k}\) | `Sizing.effects_fixed` | \(\mathbb{R}\) | varies | Fixed investment cost (one-time, sized) |
| \(\gamma^{\text{build}}_{f,k}\) | `Investment.effects_per_size_at_build` | \(\mathbb{R}\) | varies | Per-size CAPEX charged in the build period |
| \(\phi^{\text{build}}_{f,k}\) | `Investment.effects_fixed_at_build` | \(\mathbb{R}\) | varies | Fixed CAPEX charged in the build period |
| \(\gamma^{\text{rec}}_{f,k}\) | `Investment.effects_per_size_recurring` | \(\mathbb{R}\) | varies | Recurring per-size cost (each active period) |
| \(\phi^{\text{rec}}_{f,k}\) | `Investment.effects_fixed_recurring` | \(\mathbb{R}\) | varies | Recurring fixed cost (each active period) |
| \(D^{\text{up,min}}\) | `Status.min_uptime` | \(\geq 0\) | h | Minimum consecutive uptime |
| \(D^{\text{up,max}}\) | `Status.max_uptime` | \(\geq 0\) | h | Maximum consecutive uptime |
| \(D^{\text{down,min}}\) | `Status.min_downtime` | \(\geq 0\) | h | Minimum consecutive downtime |
| \(D^{\text{down,max}}\) | `Status.max_downtime` | \(\geq 0\) | h | Maximum consecutive downtime |
| \(r_{f,k,t}\) | `Status.effects_per_running_hour` | \(\mathbb{R}\) | varies | Running cost coefficient |
| \(u_{f,k,t}\) | `Status.effects_per_startup` | \(\mathbb{R}\) | varies | Startup cost coefficient |
| \(w_t\) | weights | \(> 0\) | — | Timestep weight |
| \(\Delta t_t\) | dt | \(> 0\) | h | Timestep duration |
| \(\omega_p\) | `Dims.period_weights` | \(> 0\) | — | Global period weight (multi-period only) |
| \(\omega_{k,p}\) | `Effect.period_weights` | \(> 0\) | — | Per-effect period weight |

## Naming Conventions

| Convention | Meaning | Example |
|---|---|---|
| Uppercase Latin | Decision variables | \(P\) (power/flow rate), \(E\) (stored energy) |
| Lowercase Latin | Relative/dimensionless parameters | \(\underline{p}\) (rel. min), \(\bar{p}\) (rel. max) |
| Greek | Physical properties | \(\eta\) (efficiency), \(\delta\) (loss rate) |
| Overbar / underbar | Bounds | \(\bar{P}\) (capacity), \(\underline{P}\) (lower bound) |
| Subscripts | Indexing | \(f\) (flow), \(t\) (time), \(s\) (storage), \(b\) (bus), \(k\) (effect), \(j\) (source effect) |
| Superscripts | Qualification | \(\eta^{\text{c}}\) (charge), \(\eta^{\text{d}}\) (discharge) |
