# ModelData as a DataTree — long-form effect storage

**Status:** Draft / design — follow-up to #217 (step 1 landed as #220).
**Scope:** the internal ModelData layer only. Elements API and Model
(linopy) semantics are unchanged; this is about how model data is stored,
validated, and serialized between them.

---

## 1. Problem

Four structural limits of the current design (a dataclass of DataArrays per
table, serialized as one `xr.Dataset` per netCDF group):

1. **Dense by default.** #220 fixed `effect_coeff` (one stacked row per
   (flow, effect) pair instead of a zeros cross-product), but ~10 more
   effect-keyed families are still dense over `effect` — and often over
   `time`/`period` too: `sizing_effects_*`, `invest_effects_*` (×4),
   `status_effects_*`, `cstatus_effects_*`, and the storage analogues.
2. **One Dataset = one size per dim name.** Stacking the remaining families
   the #220 way forces a fresh dim name per family (`contribution`,
   `contribution_t`, `sizing_contribution`, …) because differently-sized
   stacked tables cannot share a dim name within one Dataset. Same for the
   labeling coords (`contribution_flow`, …) — prefixes exist only to avoid
   collisions with the real `flow`/`effect` dims in the merged Dataset.
3. **No per-row dim heterogeneity.** A single DataArray gives every row the
   same dims, so a scalar coefficient stored next to a time-varying one
   carries a full `(time[, period])` envelope of copies. #220 kept this: a
   scalar `0.04` still occupies 8760 floats per row.
4. **IO is a hand-rolled tree.** `ModelData.to_netcdf` already writes one
   group per table plus `model/meta` — manually reimplementing what
   `xr.DataTree` does natively.

Downstream symptoms of the same disease:

- `model.py` needs `(x != 0).any()` runtime scans per dense family to skip
  empty work; sparse storage makes presence itself the signal.
- `contributions.py` reconstructs a dense `(contributor, effect, time)`
  array post-solve and `Result.save()` serializes it into every results
  file — mostly zeros.

## 2. Design

### 2.1 Container: `xr.DataTree`

```
DataTree
├── meta                    # dims, dt, weights — today's model/meta
├── flows                   # per-flow arrays: (flow, …) as today
│   └── effect_coeff        # effect ledger node (§2.2)
│       ├── scalar          # value (contribution,)
│       ├── time            # value (contribution, time)
│       └── time_period     # value (contribution, time, period)
├── buses
├── converters
├── effects
└── storages
```

Properties this buys:

- **Node-scoped dims.** Every node is its own Dataset: `contribution` can
  have a different size in every node, and labeling coords can use plain
  names (`flow`, `effect`) because they no longer merge into a Dataset that
  owns real `flow`/`effect` dims. The prefix mangling from #220
  (`contribution_flow`) disappears.
- **Native IO.** `tree.to_netcdf(path)` / `xr.open_datatree(path)` replace
  the manual group loop. Groups nest arbitrarily deep, so signature
  children serialize for free.
- **Same xarray idioms.** Nodes hold ordinary Datasets; model building and
  results code keep working with DataArrays.

### 2.2 Signature grammar — long form from the start

Coefficient ingestion classifies each input (`Variate`) by its **natural
dims signature** — `()`, `(time,)`, `(period,)`, `(time, period)` — as
given by the user, with no broadcasting. Rows of the same signature stack
into one array; each signature is a child node:

- Child nodes are **discovered from the data**: a model with only scalar
  coefficients has only the `scalar` child. This generalizes a
  static/dynamic split to N signatures without hardcoding a dichotomy and
  without parallel field names.
- Within every child, the row dim is uniformly named `contribution`, with
  non-dim labeling coords `flow` and `effect`. The dim carries no index
  coord, so it can never participate in alignment.
- Absent child = no rows of that shape. No `(x != 0).any()` scans; no NaN
  or zero sentinels.
- **Memory:** a scalar coefficient costs 8 bytes from the moment of
  ingestion. Dresden-scale (60 flows × 15 effects × 8760 h, ~156 pairs,
  mostly scalar): dense ≈ 63 MB → #220 stacked ≈ 11 MB → signature nodes
  ≈ **tens of KB** plus only the genuinely time-varying rows.

Validation at build: (flow, effect) pairs must be unique across *all*
signature children of a family node.

### 2.3 The ledger end-state

Every effect-keyed family is the same shape: a coefficient table pairing a
**contributor** and an **effect** on a **channel**, multiplied by one solver
variable:

| channel (node)        | today's field                      | variable            |
|-----------------------|------------------------------------|---------------------|
| `flow_hour`           | `effect_coeff`                     | `flow--rate · dt`   |
| `running_hour`        | `status_effects_running`           | `flow--on · dt`     |
| `startup`             | `status_effects_startup`           | `flow--startup`     |
| `running_hour` (comp) | `cstatus_effects_running`          | `component--on · dt`|
| `per_size`            | `sizing_effects_per_size`          | `flow--size`        |
| `fixed`               | `sizing_effects_fixed`             | indicator / const   |
| `per_size_at_build`   | `invest_effects_per_size_at_build` | `invest--size_at_build` |
| …                     | …                                  | …                   |

Under the tree, each channel is a node with signature children — the ten
dataclass fields become one uniform subtree. `_create_effects` collapses
into a loop over channels × signatures (loops over *structure*, a handful
of iterations — not loops over coordinates), each iteration running the
same select → groupby → reindex pattern proven in #220.
`contributions.py` becomes a re-aggregation of the same ledger against
solved values instead of a parallel reimplementation.

This layout is also the working answer to PyPSA #1788's open schema
question ("how are effect-keyed coefficients stored on components") and
matches the #1473 segment-index device — keeping fluxopt the reference
implementation for the Effects-in-core push.

### 2.4 Model building pattern

```python
def _channel_temporal(self, node: xr.DataTree, var, scale) -> Any:
    """Σ_rows coeff · var[flow(row)] · scale, grouped onto the effect dim."""
    expr = 0
    for child in node.children.values():
        coeff = child['value']
        pair_flow = xr.DataArray(child['flow'].values, dims=['contribution'])
        pair_effect = xr.DataArray(child['effect'].values, dims=['contribution'], name='effect')
        expr = expr + (
            (var.sel(flow=pair_flow) * (coeff * scale).drop_vars(['flow', 'effect']))
            .groupby(pair_effect)
            .sum()
            .reindex(effect=self.effect_index)
            .drop_vars('flow', errors='ignore')
        )
    return expr
```

All linopy-native (vectorized `sel`, `groupby`, `reindex` — no custom
sparse helpers), broadcasting the scalar signature against `dt`/time only
inside the solver expression, where that size is inherent to the model.

### 2.5 Results side

- Contribution breakdowns are computed from the ledger and kept **stacked**;
  the dense `(contributor, effect, time)` view becomes an on-demand
  accessor, not a stored/serialized artifact. Results files stop carrying
  dense zeros.

## 3. Migration plan

- **Phase 0** — done: #220 (stacked `effect_coeff`, single array).
- **Phase 1 — container swap.** `ModelData` holds a `DataTree`; the table
  dataclasses become thin typed views over nodes (construction + validation
  keep living there); IO becomes `to_netcdf`/`open_datatree`. No layout
  changes inside tables yet. Pure refactor, bit-identical solves.
- **Phase 2 — `effect_coeff` → signature node.** Drop the per-row envelope;
  plain `flow`/`effect` coord names. Model/contributions consume children.
- **Phase 3 — remaining families.** Channel nodes for status / sizing /
  investment / storage effects; `_create_effects` becomes the channel loop;
  delete the `(x != 0).any()` guards.
- **Phase 4 — results.** Stacked contributions, on-demand dense view, slim
  results files.

Each phase is a separate PR with the full suite green; pre-1.0, so netCDF
layout changes need no back-compat shims — but Phase 1 should land before
any stored-model corpus exists.

Coordination: rebase over #213 (flat time axis) — orthogonal (signature
envelopes simply use the flat `time` dim and `time_period` coord).

## 4. Alternatives considered

- **Twin static/`_t` fields per family** (PyPSA's split, transposed):
  works, but doubles field count, forces per-family dim-name mangling, and
  hardcodes a two-signature dichotomy that `(period,)`-varying inputs
  already break. Rejected in favor of discovered signatures.
- **One variable per (flow, effect) pair**, each with exactly its dims:
  conceptually purest, but a large model yields Datasets with thousands of
  variables — per-variable overhead in xarray and netCDF makes this slow.
  Signature grouping keeps arrays few and fat.
- **pandas long-form table** (true tidy format): loses xarray alignment,
  broadcasting, and coords-carrying netCDF IO; would reintroduce a
  conversion layer at the linopy boundary.
- **`sparse`-backed DataArrays** (pydata/sparse COO under xarray): keeps
  dense *semantics* with sparse storage, but linopy interop is unproven and
  fill-value edge cases (NaN vs 0 sentinels) are exactly the bugs the
  explicit layout avoids.

## 5. Open questions

1. **Facade depth.** Do the table dataclasses survive as typed views
   (recommended: yes — they carry validation and editor affordances), or is
   the tree accessed raw? If they survive, do `to_dataset`/`from_dataset`
   remain public?
2. **Node schema registry.** Field-comment schema doesn't scale to a tree.
   Introduce a small registry (name → dims, labeling coords, doc) that
   drives construction, validation, and docs — the PyPSA `component_attrs`
   idea at tree granularity?
3. **Converter coefficients.** `converters.flow_coeff` is dense
   `(converter, eq_idx, flow, time)` and feeds `sparse_weighted_sum` — same
   disease, same cure, but out of scope here. When linopy natively covers
   grouped sparse sums, both paths converge.
4. **xarray floor.** `DataTree` is stable since 2024.10; pin the minimum
   xarray accordingly.
5. **Signature canon.** Fixed child-name set (`scalar`, `time`, `period`,
   `time_period`) vs derived from dims — fixed set is greppable and
   self-documenting; derived is future-proof for `scenario`. Start fixed,
   derive later if a new dim arrives.
