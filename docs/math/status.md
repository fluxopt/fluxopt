# Status (On/Off Constraints)

## Overview

Status adds binary on/off behavior to flows. The core variables are a binary
status indicator and transition indicators (startup/shutdown). Duration tracking
variables enforce minimum and maximum consecutive up- and downtime.

## Variables

| Symbol | Code | Domain | Description |
|---|---|---|---|
| \(\sigma_{f,t}\) | `flow--on[flow, time]` | \(\{0, 1\}\) | On/off indicator |
| \(\tau^+_{f,t}\) | `flow--startup[flow, time]` | \(\{0, 1\}\) | Startup event indicator |
| \(\tau^-_{f,t}\) | `flow--shutdown[flow, time]` | \(\{0, 1\}\) | Shutdown event indicator |
| \(\mathrm{D}^{\text{up}}_{f,t}\) | `uptime[flow, time]` | \(\geq 0\) | Consecutive uptime [h] |
| \(\mathrm{D}^{\text{down}}_{f,t}\) | `downtime[flow, time]` | \(\geq 0\) | Consecutive downtime [h] |

## Semi-Continuous Flow Rates

With fixed size, the on/off indicator gates the flow rate bounds:

\[
\bar{\mathrm{P}}_f \cdot \underline{\mathrm{p}}_{f,t} \cdot \sigma_{f,t} \leq P_{f,t} \leq \bar{\mathrm{P}}_f \cdot \bar{\mathrm{p}}_{f,t} \cdot \sigma_{f,t}
\]

When \(\sigma_{f,t} = 0\): \(P_{f,t} = 0\). When \(\sigma_{f,t} = 1\):
\(P_{f,t} \in [\bar{\mathrm{P}}_f \underline{\mathrm{p}}, \bar{\mathrm{P}}_f \bar{\mathrm{p}}]\).

This gives the semi-continuous behavior \(\{0\} \cup [\underline{\mathrm{P}}, \bar{\mathrm{P}}]\).

## Switch Transitions

Startup and shutdown indicators are linked to status changes:

\[
\tau^+_{f,t} - \tau^-_{f,t} = \sigma_{f,t} - \sigma_{f,t-1} \quad \forall \, t > 0
\]

At the first timestep with known previous state \(\sigma_{f,0}^{\text{prev}}\):

\[
\tau^+_{f,0} - \tau^-_{f,0} = \sigma_{f,0} - \sigma_{f,0}^{\text{prev}}
\]

The previous state is derived from `Flow.prior`: on if the last prior value > 0,
off otherwise. Without `prior`, the initial transition is unconstrained.

## Duration Tracking

Duration tracking uses a Big-M formulation to count consecutive hours in a state.

### Uptime

The uptime variable \(\mathrm{D}^{\text{up}}_{f,t}\) tracks consecutive on-hours:

**Reset when off:**

\[
\mathrm{D}^{\text{up}}_{f,t} \leq \sigma_{f,t} \cdot M \quad \forall \, t
\]

**Forward accumulation:**

\[
\mathrm{D}^{\text{up}}_{f,t+1} \leq \mathrm{D}^{\text{up}}_{f,t} + \Delta t_t \quad \forall \, t
\]

**Backward tightening (force accumulation when on):**

\[
\mathrm{D}^{\text{up}}_{f,t+1} \geq \mathrm{D}^{\text{up}}_{f,t} + \Delta t_t + (\sigma_{f,t+1} - 1) \cdot M \quad \forall \, t
\]

where \(M\) is the total horizon length (Big-M constant).

### Downtime

Downtime tracking uses the same formulation applied to the inverted state
\((1 - \sigma_{f,t})\).

### Minimum Duration

Minimum uptime is enforced at shutdown transitions — the accumulated duration
must meet the minimum before turning off:

\[
\mathrm{D}^{\text{up}}_{f,t} \geq \mathrm{D}^{\text{up,min}} \cdot (\sigma_{f,t} - \sigma_{f,t+1}) \quad \forall \, t < |\mathcal{T}|
\]

The term \((\sigma_{f,t} - \sigma_{f,t+1})\) equals 1 only at shutdown
(on → off), enforcing \(\mathrm{D}^{\text{up}}_{f,t} \geq \mathrm{D}^{\text{up,min}}\).

Minimum downtime follows the same pattern on \((1 - \sigma)\).

### Maximum Duration

Maximum duration is enforced as an upper bound on the duration variable itself.

Duration values are in **hours**. With sub-hourly timesteps (e.g., `dt=0.5`),
a `min_uptime=2` means the unit must stay on for 4 consecutive timesteps.

### Previous Duration Carryover

`Flow.prior_rates` provides the flow rates from timesteps **before** the
optimization horizon. This lets the solver know the initial on/off state
and how long the unit has been running or idle:

The prior rates determine:

1. **Initial on/off state**: last value > 0 means on, = 0 means off
2. **Previous duration**: consecutive hours in the current state at the end
   of the prior, used for duration constraint carryover

When prior provides historical state, the previous duration is computed
by counting consecutive matching timesteps at the end of the prior. At \(t=0\):

\[
\mathrm{D}^{\text{up}}_{f,0} = \sigma_{f,0} \cdot (\mathrm{D}^{\text{up,prev}} + \Delta t_0)
\]

If the previous uptime hasn't yet met the minimum, the unit is forced to stay on:

\[
\sigma_{f,0} \geq 1 \quad \text{if } 0 < \mathrm{D}^{\text{up,prev}} < \mathrm{D}^{\text{up,min}}
\]

## Effect Contributions

### Running Costs

A per-hour cost while the unit is on, independent of the flow rate:

\[
\Phi_{k,t}^{\text{running}} = \sum_{f} \mathrm{r}_{f,k,t} \cdot \sigma_{f,t} \cdot \Delta t_t
\]

where \(\mathrm{r}_{f,k,t}\) is `Status.effects_per_running_hour[k]`.

### Startup Costs

A one-time cost charged each time the unit switches from off to on:

\[
\Phi_{k,t}^{\text{startup}} = \sum_{f} \mathrm{u}_{f,k,t} \cdot \tau^+_{f,t}
\]

where \(\mathrm{u}_{f,k,t}\) is `Status.effects_per_startup[k]`.

Both feed into the [per-timestep effect equation](effects.md).

## Interaction with Sizing

When a flow has both `Status` and `Sizing`, the on/off indicator and the size
variable are decoupled via Big-M constraints:

\[
P_{f,t} \leq \sigma_{f,t} \cdot M^+ \qquad \text{(on-indicator gates flow)}
\]

\[
P_{f,t} \leq S_f \cdot \bar{\mathrm{p}}_{f,t} \qquad \text{(rate limited by size)}
\]

\[
P_{f,t} \geq (\sigma_{f,t} - 1) \cdot M^- + S_f \cdot \underline{\mathrm{p}}_{f,t} \qquad \text{(minimum when on)}
\]

where \(M^+ = \mathrm{S}^+ \cdot \bar{\mathrm{p}}_{f,t}\) and \(M^- = \mathrm{S}^+ \cdot \underline{\mathrm{p}}_{f,t}\),
using the maximum possible size \(\mathrm{S}^+\) as Big-M.

An additional constraint prevents the unit from being "on" with zero size:

\[
\sigma_{f,t} \leq S_f \quad \forall \, t
\]

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\sigma_{f,t}\) | On/off binary | `flow--on[flow, time]` |
| \(\tau^+_{f,t}\) | Startup indicator | `flow--startup[flow, time]` |
| \(\tau^-_{f,t}\) | Shutdown indicator | `flow--shutdown[flow, time]` |
| \(\mathrm{D}^{\text{up}}_{f,t}\) | Consecutive uptime | `uptime[flow, time]` |
| \(\mathrm{D}^{\text{down}}_{f,t}\) | Consecutive downtime | `downtime[flow, time]` |
| \(\mathrm{D}^{\text{up,min}}\) | Minimum uptime | [`Status.min_uptime`](../api/fluxopt/elements.md#fluxopt.elements.Status(min_uptime)) |
| \(\mathrm{D}^{\text{up,max}}\) | Maximum uptime | [`Status.max_uptime`](../api/fluxopt/elements.md#fluxopt.elements.Status(max_uptime)) |
| \(\mathrm{D}^{\text{down,min}}\) | Minimum downtime | [`Status.min_downtime`](../api/fluxopt/elements.md#fluxopt.elements.Status(min_downtime)) |
| \(\mathrm{D}^{\text{down,max}}\) | Maximum downtime | [`Status.max_downtime`](../api/fluxopt/elements.md#fluxopt.elements.Status(max_downtime)) |
| \(\mathrm{r}_{f,k,t}\) | Running cost coefficient | [`Status.effects_per_running_hour`](../api/fluxopt/elements.md#fluxopt.elements.Status(effects_per_running_hour)) |
| \(\mathrm{u}_{f,k,t}\) | Startup cost coefficient | [`Status.effects_per_startup`](../api/fluxopt/elements.md#fluxopt.elements.Status(effects_per_startup)) |
| \(M\) | Big-M (horizon length) | computed |

See [Notation](notation.md) for the full symbol table.
