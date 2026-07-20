# Config direction: a validated, serializable element layer

**Status:** Draft / decision record.
**Audience:** ourselves; informs the PyPSA-core Effects push (`pypsa-core-integration.md`).
**Decision:** adopt **pydantic** for the element layer ("standalone DX first"),
with structural YAML/JSON round-trip and time-series kept as *references*. A
declarative **YAML-math** layer (linopy/Calliope) is a separate, later,
complementary concern — not a blocker.

---

## 1. Question

Should fluxopt's user-facing model definition become more config-/YAML-like —
using pydantic or similar — and can it reuse the emerging YAML-math tooling from
linopy?

Two sub-questions that turn out to live at **different layers**:

- **Data/config:** *what components exist and their parameters* → the element
  dataclasses (`elements.py`, `components.py`).
- **Math:** *the equations themselves* → the model builder (`model.py`,
  `constraints/`).

They compose; adopting one does not require the other.

## 2. Where fluxopt and PyPSA stand

fluxopt's three layers — `Elements` (dataclasses) → `ModelData` (`xr.Dataset`s)
→ `Model` (linopy) — already make the element layer *declarative value objects*.
Construction is a single `optimize()` call over lists of dataclasses, not an
imperative mutation loop. Validation is hand-rolled in `__post_init__`;
serialization is netCDF/DataTree at the `ModelData` layer, not the element layer.

PyPSA (v1.0), for contrast:

| | fluxopt | PyPSA v1.0 | PyPSA direction |
|---|---|---|---|
| Storage | dataclasses → `xr.Dataset`s → linopy | pandas frames per component + `_t` dicts | `Components` class is an **access wrapper over the same pandas frames**; default in v2.0 |
| Construction | declarative lists of dataclasses | imperative `n.add(...)` | — |
| Validation | `__post_init__` guards | CSV attribute registry (`component_attrs/*.csv`) coerced at `n.add` | vague "stricter validation" — **built on the registry, not pydantic** |
| Multi-quantity | **Effect system** (N quantities, chaining, any objective) | cost objective + CO₂ `GlobalConstraint` | "categorised components: separate physical components from shared properties" |
| Piecewise | `PiecewiseConversion` (shared weights, auto LP/SOS2) | none | #1473 → PR #1603 (open, ~v1.3): segments MultiIndex + SOS2 |

**Key finding:** core PyPSA has **no pydantic and no typed-schema plans**. Its
"schema" is the CSV registry; its declarative ambitions (below) target *math and
topology*, not typed validated objects.

## 3. What PyPSA is actually doing on "declarative" (July 2026)

Three open, maintainer-engaged proposals — none merged, none pydantic:

- **[#1789](https://github.com/PyPSA/PyPSA/issues/1789)** (@FabianHofmann) —
  reusable component *baskets* (`battery = bus+link+store` as one unit,
  `n.basket.<name>`), registered from files/dicts (likely YAML). This is the
  concrete design behind the roadmap's "categorised components."
- **[#1796](https://github.com/PyPSA/PyPSA/issues/1796)** (@brynpickering) —
  replace GlobalConstraints with **YAML text mathematics** (Calliope-style
  `parameters`/`expressions`/`constraints`). Notably flags our Effects issue
  **[#1788](https://github.com/PyPSA/PyPSA/issues/1788)** as *"potentially
  superseded."*
- **[linopy #561](https://github.com/PyPSA/linopy/issues/561)** — the text-based
  (YAML) math interface the above depend on. **Proposal only, no PR**; #1796 says
  it is "not yet capable" yet. The working reference is
  [Calliope 0.7+](https://calliope.readthedocs.io/en/latest/user_defined_math/),
  which already stores all its optimization math in YAML.

Corrected read: PyPSA *is* moving toward YAML — but at the **constraint-math** and
**component-composition** layers, not as a typed data schema. The config YAML most
people see (`config.default.yaml`) is a **pypsa-eur Snakemake workflow** config,
not the core data model.

## 4. Decision: pydantic element layer ("standalone DX first")

Optimize fluxopt's own ergonomics. Convert the element layer to pydantic:

- **Declarative validation** replaces the `__post_init__` guards (carrier-match,
  status-requires-size, breakpoint-length, …) with validators and better errors.
- **JSON Schema for free** — the real prize: a future config front-end, GUI, and
  LLM-assisted authoring all validate against one schema.
- **Structural round-trip** to dict/YAML/JSON, with time-series kept as
  references.

### 4.1 Time-series stay references, not inline

`Variate` fields are `float | series | array`. Real profiles (8760+ points,
multi-period) do **not** belong inline in YAML — that is exactly why PyPSA splits
static frames from `_t` time-varying data. Model it as:

```
Variate := float | list[float] | ProfileRef
ProfileRef := reference into a CSV/netCDF column (name + source)
```

Structural YAML carries the graph + scalars + refs; profiles live in data files.
Inlining profiles produces a toy; referencing them is real config.

### 4.2 Known friction

- pydantic + `xr.DataArray`/`Variate` needs `arbitrary_types_allowed` or a custom
  validator (bounded work).
- pydantic v2 becomes a hard runtime dep alongside the scientific stack.

### 4.3 Portability guardrail (for the Effects push)

PyPSA core is registry-based, not pydantic — so keep the pydantic layer *separable*
from the modeling semantics. In particular, keep the Effect **declaration** (unit,
coefficients) cleanly split from the **study verbs** (bound / minimize / price),
per `pypsa-core-integration.md`. The schema then stays portable even as PyPSA moves
the *constraint* side to YAML-math.

## 5. Can fluxopt reuse linopy/Calliope YAML-math?

**Not today** (linopy #561 is unimplemented), but **naturally later** — fluxopt's
backend already *is* linopy, so a YAML-math layer would be reusable at the same
point PyPSA would use it.

**Where it fits** — the linear-algebraic subset: Effect accumulation, Effect
bounds, cross-effect chains (`contribution_from`), and user custom constraints
(today the `customize` callback). This is exactly the `sum(co2, over=snapshots) <=
limit` shape Calliope/#561 handle, and exactly what #1796 targets.

**Where it does not** — the procedural parts of the builder: piecewise formulation
selection (LP tangent vs SOS2 vs incremental), status/uptime big-M, storage level
dynamics, and the materialization-closure algorithm. YAML-math expresses *an*
equation; it does not decide *which formulation to emit*. So YAML-math is a good
front-end for the effect/constraint layer, not a replacement for `constraints/`.

### 5.1 Illustrative sketch (Calliope-style, not a commitment)

An Effect total plus a total cap, expressed declaratively:

```yaml
expressions:
  effect_total:
    foreach: [effect]
    equation: >
      sum(flow_rate * effects_per_flow_hour * dt, over=[flow, time])
      + sum(size * effects_per_size, over=flow)

constraints:
  effect_total_cap:
    foreach: [effect]
    where: "total_max"          # only effects that declare a cap
    equation: "effect_total <= total_max"
```

The point: Effects **declare the quantity**; YAML-math **expresses the bound**.
They compose — which is the counter-argument to #1796's "superseded" framing:
demonstrating Effects in YAML-math idiom shows the two are complementary, not
rival. That demo is a strong card in the #1788 ↔ #1796 discussion.

## 6. Plan

1. **Don't block on YAML-math** — it is unimplemented in linopy. `constraints/`
   stays the source of truth.
2. **pydantic element layer**, structural-only round-trip, `ProfileRef` for
   time-series; port `__post_init__` guards to validators; emit JSON Schema.
3. **Design the YAML-math seam now, cheaply** — keep Effect bounds +
   `contribution_from` + `customize` expressible as vectorized linopy expressions
   so a text front-end can slot in later.
4. **Feed the sketch (§5.1) into the PyPSA discussion** as evidence Effects and
   YAML-math compose.

## 6a. Implementation status

**Phase 1 landed in this PR** — element layer on pydantic, validation + schema:

- `elements.py` / `components.py` element classes are now
  `pydantic.dataclasses.dataclass` with `arbitrary_types_allowed` (so `Variate`
  arrays / `IdList` pass through by `isinstance`). `__post_init__` guards,
  `field(init=False)`, nested-instance identity, and in-place qualification all
  survive unchanged (pydantic keeps instances — `revalidate_instances='never'`).
- Construction now validates types with pydantic errors. `ruff`'s
  `flake8-type-checking` is told `pydantic.dataclasses.dataclass` evaluates
  annotations at runtime, so annotation imports stay at runtime.
- **JSON Schema works for every element type** — `fluxopt.element_schema(Flow)`
  and `all_element_schemas()`. Better than §4.2 feared: arbitrary `Variate`
  fields degrade to permissive `{}` rather than erroring.

**Phase 2 landed (stacked PR)** — `ProfileRef` + full structural round-trip:

- **`ProfileRef`** (`float | list | ProfileRef` at last, in `types.py`): a
  serializable reference to a time-series in a data file, with `.resolve(sources)`
  → `DataArray`. Added to the `Variate` union so config round-trips without
  inlining profiles; `as_dataarray` rejects an unresolved ref loudly.
- **`to_dict` / `from_dict`** (`fluxopt.to_dict`, `fluxopt.from_dict`): JSON-safe
  round-trip for **every** element type, components included. `IdList` got a
  pydantic core schema (validate-from-list / serialize-to-list). The feared
  "non-idempotent qualification" was a **non-issue**: `Flow.__post_init__` resets
  `id = short_id`, so the parent re-qualifies cleanly on rebuild (`b(gas)`, not
  `b(b(gas))`). Inline raw-array `Variate` still doesn't serialize — that's the
  `ProfileRef` path.

**Phase 3 landed (stacked PR)** — the YAML front door + layer naming.

Philosophy **B** (chosen): layer 1 is an inert, validated *declaration*, consumed
by a builder — not a mutable domain object. This mirrors the codebase's existing
"declaration vs use" principle. The three layers, named to match:

| layer | name | role | serialization |
|---|---|---|---|
| 1 | **`FlowSystem`** | what the user authors (elements + config) | dict / YAML |
| 2 | `ModelData` | materialized xarray | netCDF |
| 3 | **`FlowSystemModel`** | the linopy solver; owns `.m` (escape hatch) | → `Result` |

- **`FlowSystem`** (`flow_system.py`): a pydantic aggregate mirroring `optimize()`;
  `from_dict`/`from_yaml`/`to_dict`/`to_yaml`; `.optimize(sources=...)` delegates to
  the existing pipeline (`ModelData.build → FlowSystemModel`). Python construction
  stays first-class — the same object is built in code or loaded from YAML.
- **`ProfileRef` auto-resolution**: `.optimize(sources=...)` runs a recursive
  pre-pass (`_resolve_refs`) that swaps every `ProfileRef` for `resolve(sources)`
  on a **deep copy**, so the `FlowSystem` stays reusable across different data.
  Sources are passed **in code** (`{id: Dataset}`), not via file paths in YAML —
  structure is declarative, data supply is explicit.
- **Naming decision**: the flagship name `FlowSystem` goes to the *user-authored*
  layer; today's solver was renamed `FlowSystem` → **`FlowSystemModel`** (it owns
  a linopy `Model` as `.m`, so `...Model` fits and there's no clash). `*Spec` was
  rejected — it decorates the primary user object; the qualifier belongs on the
  internal solver. Narrow rename, done pre-1.0.

**Still deferred (Phase 4):**

- **Porting `__post_init__` guards to pydantic validators**, and moving id
  qualification from element construction to the build step (the declaration-vs-use
  tightening). Both are incremental cleanup, not required for the win.

## 7. Non-goals

- Inlining time-series *profiles* in YAML (they belong in data files, referenced
  by `ProfileRef`). Declaring system *structure* in YAML is exactly the goal.
- A mutable domain object with `.add()`/`.remove()` (philosophy A) — `FlowSystem`
  is an inert declaration; assemble the lists, then hand them to the builder.
- Replacing `constraints/` with declarative math.
- Blocking any of the above on an unmerged linopy feature.
