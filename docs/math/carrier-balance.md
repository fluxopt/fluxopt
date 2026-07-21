# Carrier Balance

## Formulation

Every carrier \(b\) must be balanced at every timestep — total outflow equals total inflow:

\[
\sum_{f \in \mathcal{F}_b^{\text{out}}} P_{f,t} - \sum_{f \in \mathcal{F}_b^{\text{in}}} P_{f,t} = 0 \quad \forall \, b \in \mathcal{B}, \; t \in \mathcal{T}
\]

where:

- \(\mathcal{F}_b^{\text{out}}\) — flows that produce into carrier \(b\) (imports of ports and outputs of converters)
- \(\mathcal{F}_b^{\text{in}}\) — flows that consume from carrier \(b\) (exports of ports and inputs of converters)

The sign convention uses coefficients: \(+1\) for flows producing into the carrier and
\(-1\) for flows consuming from the carrier. The constraint is then:

\[
\sum_{f \in \mathcal{F}_b} \text{coeff}_{b,f} \cdot P_{f,t} = 0 \quad \forall \, b, t
\]

## Parameters

| Symbol | Description | Reference |
|---|---|---|
| \(\mathcal{F}_b^{\text{out}}\) | Flows producing into carrier \(b\) | `carrier_coeff[f.id] = +1` (port imports, converter outputs, storage discharging) |
| \(\mathcal{F}_b^{\text{in}}\) | Flows consuming from carrier \(b\) | `carrier_coeff[f.id] = -1` (port exports, converter inputs, storage charging) |
| \(P_{f,t}\) | Flow rate variable | `flow--rate[flow, time]` |
| \((b\!:\!n)\) | Compound carrier-node coordinate | e.g., `heat:A`, `heat:B` |
| \(\mathcal{F}_{b:n}\) | Flows assigned to node \(n\) of carrier \(b\) | Subset of \(\mathcal{F}_b\) with matching `node` |
| \(\text{coeff}_{b:n,f}\) | Coefficient of flow \(f\) in node \(n\)'s balance | `+1` or `-1`, same convention as single-node |

See [Notation](notation.md) for the full symbol table.

## Example

A thermal carrier with a boiler output (3 MW) and a demand input (3 MW):

\[
\underbrace{P_{\text{boilerHeat},t}}_{+1 \times 3} + \underbrace{(-1) \cdot P_{\text{demand},t}}_{-1 \times 3} = 0 \quad \checkmark
\]

## Multi-Node Carriers

A carrier can be split into independent **nodes**, each with its own balance
equation. This models spatially separated subsystems on the same physical
medium — e.g., separate heat networks in different buildings.

### Formulation

Each node \(n\) of carrier \(b\) gets its own balance constraint:

\[
\sum_{f \in \mathcal{F}_{b:n}} \text{coeff}_{b:n,f} \cdot P_{f,t} = 0 \quad \forall \, (b, n), \; t \in \mathcal{T}
\]

The compound coordinate \(b\!:\!n\) (e.g., `heat:A`, `heat:B`) identifies each
node in the carrier dimension. Flows on different nodes never interact — their
balance equations are fully independent.

### Example

Two independent heat nodes A and B, each with its own supply and demand:

\[
\begin{aligned}
\text{Node A:}\quad & P_{\text{src\_a(heat:A)},t} - P_{\text{sink\_a(heat:A)},t} = 0 \\
\text{Node B:}\quad & P_{\text{src\_b(heat:B)},t} - P_{\text{sink\_b(heat:B)},t} = 0
\end{aligned}
\]

Node A's supply serves only node A's demand. Node B is balanced independently.

### Usage

The flow id auto-includes the node: `Flow(carrier='heat', node='A')` gets `id='heat:A'`,
which qualifies to `src_a(heat:A)` after component qualification.
