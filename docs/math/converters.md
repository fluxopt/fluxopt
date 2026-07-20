# Converters

A `Converter` couples its input and output flows. fluxopt offers two
conversion modes: a **linear** form (coefficients fixed against the
operating point — they may still vary in time) and a
**piecewise-linear** form (`PiecewiseConversion`) for behaviour that
depends on the operating point itself — part-load efficiency curves,
load-dependent heat-to-power ratios.

## Linear Conversion

### Formulation

Each conversion equation enforces a linear coupling between flows:

\[
\sum_{f} \mathrm{a}_{f,i} \cdot P_{f,t} = 0 \quad \forall \, \text{converter}, \; i, \; t \in \mathcal{T}
\]

where \(\mathrm{a}_{f,i}\) is the conversion coefficient for flow \(f\) in equation \(i\).
A converter can have multiple equations (one per row in `conversion_factors`),
allowing multi-output devices like CHP plants. Coefficients may also broadcast
over \(t\) in the API — see [Indexing Convention](notation.md#indexing-convention).

### Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\mathrm{a}_{f,i}\) | Conversion coefficient | [`Converter.conversion_factors`](../api/fluxopt/components.md#fluxopt.components.Converter.conversion_factors) |
| \(i\) | Equation index within a converter | row in `conversion_factors` |
| \(P_{f,t}\) | Flow rate variable | `flow--rate[flow, time]` |

See [Notation](notation.md) for the full symbol table.

### Examples

#### Boiler

A gas boiler with thermal efficiency \(\eta_{\text{th}} = 0.9\):

\[
\eta_{\text{th}} \cdot P_{\text{gas},t} - P_{\text{th},t} = 0
\]

\[
0.9 \cdot P_{\text{gas},t} = P_{\text{th},t}
\]

So 10 MW gas input produces 9 MW thermal output.

#### Power-to-Heat

An electric resistance heater with efficiency \(\eta = 0.99\):

\[
\eta \cdot P_{\text{el},t} - P_{\text{th},t} = 0
\]

#### Heat Pump

A heat pump with COP = 3.5 has **two** conversion equations — COP definition
and energy balance:

\[
\text{COP} \cdot P_{\text{el},t} - P_{\text{th},t} = 0
\]

\[
P_{\text{el},t} + P_{\text{src},t} - P_{\text{th},t} = 0
\]

So 1 MW electrical input draws 2.5 MW from the environment and produces
3.5 MW thermal output.

#### CHP (Combined Heat and Power)

A CHP with \(\eta_{\text{el}} = 0.4\) and \(\eta_{\text{th}} = 0.5\) has **two**
conversion equations:

\[
\eta_{\text{el}} \cdot P_{\text{fuel},t} - P_{\text{el},t} = 0
\]

\[
\eta_{\text{th}} \cdot P_{\text{fuel},t} - P_{\text{th},t} = 0
\]

So 10 MW fuel input produces 4 MW electrical + 5 MW thermal.

## Piecewise Conversion

When efficiency or coupling depends on the **operating point itself** —
part-load curves, load-dependent heat-to-power ratios — set
`conversion=PiecewiseConversion(...)` on the `Converter` instead of
`conversion_factors`. (For coefficients that vary with *time but not load*,
use the linear form with time-varying `conversion_factors`.)

### Formulation

A `PiecewiseConversion` defines breakpoints \(\mathrm{b}_{f,k}\) for each flow \(f\) at \(K\)
piece-vertices \(k = 0, \dots, K-1\). At every timestep, a vector of
non-negative interpolation weights \(\lambda_{k,t}\) selects the operating
point on the curve:

\[
\sum_{k} \lambda_{k,t} = 1, \qquad \lambda_{k,t} \ge 0
\]

with at most two adjacent weights non-zero (SOS2 condition for a contiguous
curve, enforced by `linopy.add_piecewise_formulation`).

For each curve flow \(f\), the rate is the corresponding weighted breakpoint
sum:

\[
P_{f,t} \; \diamond_f \; \sum_{k} \lambda_{k,t} \cdot \mathrm{b}_{f,k}
\]

where the relation \(\diamond_f \in \{=, \le, \ge\}\) is set per flow via the
optional third tuple element. The default is equality (`==`); at most one
flow may carry an inequality sign, and only with exactly two flows.

### Methods

`linopy` auto-dispatches the formulation:

| Method | When | Aux variables |
|---|---|---|
| `lp` | Two flows, one bounded, matching-curvature curve | None — pure tangent-line constraints |
| `incremental` | Strictly monotonic breakpoints | One binary per piece |
| `sos2` | Otherwise | One \(\lambda\) per breakpoint, SOS2 |

Override with `method="sos2"` / `"incremental"` / `"lp"` if needed.

### Status gating

When `PiecewiseConversion.status` is set, the curve is gated by the converter's
on/off binary \(\sigma_{c,t}\) (see [Status](status.md)) passed as `active=` to
the linopy formulation:

- All-equality curves: \(\sigma_{c,t} = 0\) forces every \(\lambda\) to zero,
  which pins all curve flows to \(\mathrm{b}_{f,0}\) (typically zero).
- Inequality curves: \(\sigma_{c,t} = 0\) drives the bounded side to zero; the
  output's own lower bound (default \(P_f \ge 0\)) closes the loop for
  non-negative outputs.

### Availability

A separate envelope constraint scales the upper breakpoint by a per-timestep
availability \(\alpha_t \in [0, 1]\):

\[
P_{f^{\star},t} \le \alpha_t \cdot \mathrm{b}_{f^{\star},K-1} \cdot \sigma_{c,t}
\]

where \(f^{\star}\) is the first flow in the curve.

### Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\mathrm{b}_{f,k}\) | Breakpoint values per flow | [`PiecewiseConversion.points`](../api/fluxopt/elements.md#fluxopt.elements.PiecewiseConversion.points) |
| \(\lambda_{k,t}\) | Interpolation weights | linopy auxiliaries |
| \(\diamond_f\) | Curve relation | tuple bound `'=='` / `'<='` / `'>='` |
| \(\sigma_{c,t}\) | On/off binary | [`PiecewiseConversion.status`](../api/fluxopt/elements.md#fluxopt.elements.PiecewiseConversion.status) |
| \(\alpha_t\) | Availability scaling | [`PiecewiseConversion.availability`](../api/fluxopt/elements.md#fluxopt.elements.PiecewiseConversion.availability) |
