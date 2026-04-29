# Linear Conversion

## Formulation

A `Converter` enforces linear coupling between its input and output flows.
Each conversion equation requires:

\[
\sum_{f} a_{f} \cdot P_{f,t} = 0 \quad \forall \, \text{converter}, \; \text{eq\_idx}, \; t \in \mathcal{T}
\]

where \(a_f\) is the conversion coefficient for flow \(f\). A converter can have
multiple equations (one per row in `conversion_factors`), allowing multi-output
devices like CHP plants.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(a_f\) | Conversion coefficient | `Converter.conversion_factors` |
| \(P_{f,t}\) | Flow rate variable | `flow_rate[flow, time]` |

See [Notation](notation.md) for the full symbol table.

## Examples

### Boiler

A gas boiler with thermal efficiency \(\eta_{\text{th}} = 0.9\):

\[
\eta_{\text{th}} \cdot P_{\text{gas},t} - P_{\text{th},t} = 0
\]

\[
0.9 \cdot P_{\text{gas},t} = P_{\text{th},t}
\]

So 10 MW gas input produces 9 MW thermal output.

```python
Converter.boiler("boiler", thermal_efficiency=0.9, fuel_flow=gas, thermal_flow=th)
# conversion_factors = [{gas.id: 0.9, th.id: -1}]
```

### Power-to-Heat

An electric resistance heater with efficiency \(\eta = 0.99\):

\[
\eta \cdot P_{\text{el},t} - P_{\text{th},t} = 0
\]

```python
Converter.power2heat("p2h", efficiency=0.99, electrical_flow=el, thermal_flow=th)
# conversion_factors = [{el.id: 0.99, th.id: -1}]
```

### Heat Pump

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

```python
Converter.heat_pump("hp", cop=3.5, electrical_flow=el, source_flow=src, thermal_flow=th)
# conversion_factors = [{el.id: 3.5, th.id: -1}, {el.id: 1, src.id: 1, th.id: -1}]
```

### CHP (Combined Heat and Power)

A CHP with \(\eta_{\text{el}} = 0.4\) and \(\eta_{\text{th}} = 0.5\) has **two**
conversion equations:

\[
\eta_{\text{el}} \cdot P_{\text{fuel},t} - P_{\text{el},t} = 0
\]

\[
\eta_{\text{th}} \cdot P_{\text{fuel},t} - P_{\text{th},t} = 0
\]

So 10 MW fuel input produces 4 MW electrical + 5 MW thermal.

```python
Converter.chp("chp", eta_el=0.4, eta_th=0.5,
                     fuel_flow=fuel, electrical_flow=el, thermal_flow=th)
# conversion_factors = [
#     {fuel.id: 0.4, el.id: -1},
#     {fuel.id: 0.5, th.id: -1},
# ]
```

### Custom Equations

For devices not covered by factory methods, pass `conversion_factors` directly.
Each dict in the list is one equation, mapping flow short ids to coefficients:

```python
in1 = Flow('a', size=100)
in2 = Flow('b', size=100)
out = Flow('c', size=100)

conv = Converter(
    id='custom',
    inputs=[in1, in2],
    outputs=[out],
    conversion_factors=[{'a': 0.5, 'b': 0.3, 'c': -1}],
)
```

This enforces: \(0.5 \cdot P_a + 0.3 \cdot P_b - P_c = 0\).

### Time-Varying Coefficients

Conversion coefficients can be time-varying (e.g., a heat pump with hourly COP from
weather data). Pass a list or array instead of a scalar:

```python
cop_profile = [3.2, 3.5, 3.8, 3.1]  # one value per timestep
Converter.heat_pump("hp", cop=cop_profile, electrical_flow=el, source_flow=src, thermal_flow=th)
```

# Piecewise Conversion

For non-linear efficiency curves, varying COP, part-load behaviour, or
combined-heat-and-power with non-constant ratios, set `conversion=PiecewiseConversion(...)`
on the `Converter` instead of `conversion_factors`.

## Formulation

A `PiecewiseConversion` defines breakpoints \(b_{f,k}\) for each flow \(f\) at \(K\)
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
P_{f,t} \; \diamond_f \; \sum_{k} \lambda_{k,t} \cdot b_{f,k}
\]

where the relation \(\diamond_f \in \{=, \le, \ge\}\) is set per flow via the
optional third tuple element. The default is equality (`==`); at most one
flow may carry an inequality sign, and only with exactly two flows.

## Methods

`linopy` auto-dispatches the formulation:

| Method | When | Aux variables |
|---|---|---|
| `lp` | Two flows, one bounded, matching-curvature curve | None — pure tangent-line constraints |
| `incremental` | Strictly monotonic breakpoints | One binary per piece |
| `sos2` | Otherwise | One \(\lambda\) per breakpoint, SOS2 |

Override with `method="sos2"` / `"incremental"` / `"lp"` if needed.

## Status gating

When `PiecewiseConversion.status` is set, the curve is gated by a binary
\(\delta_{c,t}\) (see [Status](status.md)) passed as `active=` to the linopy
formulation:

- All-equality curves: \(\delta = 0\) forces every \(\lambda\) to zero, which
  pins all curve flows to \(b_{f,0}\) (typically zero).
- Inequality curves: \(\delta = 0\) drives the bounded side to zero; the
  output's own lower bound (default \(P_f \ge 0\)) closes the loop for
  non-negative outputs.

## Availability

A separate envelope constraint scales the upper breakpoint by a per-timestep
availability \(\alpha_t \in [0, 1]\):

\[
P_{f^{\star},t} \le \alpha_t \cdot b_{f^{\star},K-1} \cdot \delta_{c,t}
\]

where \(f^{\star}\) is the first flow in the curve.

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(b_{f,k}\) | Breakpoint values per flow | `PiecewiseConversion.points` |
| \(\lambda_{k,t}\) | Interpolation weights | linopy auxiliaries |
| \(\diamond_f\) | Curve relation | tuple bound `'=='` / `'<='` / `'>='` |
| \(\delta_{c,t}\) | On/off binary | `PiecewiseConversion.status` |
| \(\alpha_t\) | Availability scaling | `PiecewiseConversion.availability` |

## Examples

### Boiler with part-load efficiency

A gas boiler runs at 90% efficiency between 0 and 50 MW (slope 0.9), then drops
to 50% efficiency from 50 to 100 MW (slope 0.5):

```python
Converter(
    'Boiler',
    inputs=[Flow('Gas', short_id='fuel')],
    outputs=[Flow('Heat', size=100)],
    conversion=PiecewiseConversion({
        'fuel': [0, 50, 100],
        'Heat': [0, 45, 70],
    }),
)
```

### CHP with joint N-flow curve

A CHP plant with three flows linked by shared interpolation weights — every
operating point lies on the same piece of the curve:

```python
Converter(
    'CHP',
    inputs=[Flow('Gas', short_id='fuel')],
    outputs=[Flow('Power', size=100), Flow('Heat', size=100)],
    conversion=PiecewiseConversion({
        'fuel':  [0, 30, 60, 100],
        'Power': [0, 10, 22,  40],
        'Heat':  [0, 15, 30,  45],
    }),
)
```

### Convex curve via LP fast path

A monotonic-convex fuel curve (efficiency drops at high load) with an
inequality bound — solver picks `method='lp'` automatically and uses pure
tangent-line constraints (no SOS2 binaries):

```python
Converter(
    'Boiler',
    inputs=[Flow('Gas', short_id='fuel')],
    outputs=[Flow('Heat', size=100)],
    conversion=PiecewiseConversion(
        [
            ('Heat', [0, 30, 60, 100]),
            ('fuel', [0, 36, 84, 170], '>='),
        ],
        method='auto',
    ),
)
```

### With on/off and startup costs

```python
Converter(
    'Boiler',
    inputs=[Flow('Gas', short_id='fuel')],
    outputs=[Flow('Heat', size=100)],
    conversion=PiecewiseConversion(
        {'fuel': [0, 50, 100], 'Heat': [0, 45, 70]},
        status=Status(min_uptime=3, effects_per_startup={'cost': 50}),
    ),
)
```
