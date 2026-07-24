# Bringing fluxopt's modeling paradigm into PyPSA core

**Status:** Draft / requirements — intended as a handoff brief for a follow-up
design pass (fable).
**Audience:** ourselves first; ultimately PyPSA maintainers via a GitHub
discussion / enhancement proposal.
**Goal:** land fluxopt's distinctive modeling capabilities in **PyPSA core**,
natively — not as an `extra_functionality` bolt-on, not as a fork.

---


## 1. Thesis

PyPSA is excellent at network + dispatch + investment *at scale* (continent-scale
continuous LPs, PyPSA-Eur). It is deliberately lean on two things that the
oemof / flixOpt / fluxopt paradigm treats as first-class:

1. **Rich, multi-quantity accounting** — tracking many named quantities (cost,
   CO₂, primary energy, land use, …), letting *any one* be the objective and the
   rest be budgeted, with quantities that feed into each other (CO₂ → cost).
2. **Per-unit MILP detail** — part-load efficiency curves, and discrete
   "build this one unit, decide *when*" investment.

The key claim of this document: **these are not alien to PyPSA — they are
generalizations of mechanisms PyPSA already has.** If we frame the contribution
that way, core acceptance is plausible. If we frame it as a parallel subsystem,
it will (rightly) be rejected.

The load-bearing insight:

> PyPSA already has a single-quantity, single-objective, single-cap version of
> the Effect system. It's spelled `marginal_cost` / `capital_cost` (the cost
> objective) plus the CO₂ `GlobalConstraint` (a secondary quantity, accumulated
> from a per-carrier attribute, with a cap). The **Effect system is the natural
> N-quantity generalization**, with today's behavior as the default special
> case.

---

## 2. What we want to bring in

**This push is about (A), the Effect system, only.** (B) piecewise and (C)
discrete single-unit investment are documented below for context but are
explicitly **out of scope for this effort** — deferred to later, independent
pushes. We land the Effect generalization first; everything else waits.

### A. The Effect system (multi-quantity accounting)  ⭐ core value

**What it is (fluxopt today):** an `Effect` is a named tracked quantity with a
unit. Component activity contributes to effects via coefficients:

- per flow-hour (e.g. €/MWh, kg/MWh)
- per unit size / fixed, when built (investment)
- per running hour / per startup (status)

Exactly one effect is designated the **objective**; any effect can carry bounds:

- total across the horizon (weighted across periods)
- per period
- per hour (a rate, scaled by Δt)

Effects can **chain**: `contribution_from = {co2: 60}` routes an effect's value
into another (carbon price CO₂ → cost). All couplings are linear.

**Why it matters:** emission budgets, multi-criteria analysis, carbon pricing,
and "minimize CO₂ subject to a cost cap" all fall out of one uniform mechanism
instead of ad-hoc constraints.

**How it maps to PyPSA natively (the pitch):**

| PyPSA today | Generalized as |
|---|---|
| scalar objective = Σ capital + marginal cost | objective = the *designated* effect |
| `marginal_cost` column | contribution to the `cost` effect per flow-hour |
| `capital_cost` column | contribution to `cost` per unit `p_nom` |
| `Carrier.co2_emissions` + CO₂ `GlobalConstraint` | `co2` effect (accounting) **+** an `effect_limit` GlobalConstraint (the cap) |
| `GlobalConstraint` (limit) | **stays** a GlobalConstraint — new `effect_limit` type, sibling of `primary_energy`; sparse, dual `mu` free |

So: add an **`Effect` component table** (parallel to `Carrier`), let components
carry **effect-keyed coefficients** (stored with an extra index level — the same
data-model device #1473 uses for segments), and make the objective a *reference
to one effect*. Backward compat: `marginal_cost` / `capital_cost` become sugar
that populates the built-in `cost` effect; the CO₂ GlobalConstraint stays a
GlobalConstraint — now recognizable as an `effect_limit` on the built-in `co2`
effect. **Nothing breaks; existing models are the default single-effect case.**

**Cheap by construction — only materialize what constrains the solve.**
An effect matters to the *optimization* only if it is the objective or carries a
bound. An effect that is neither is a pure linear function of the solution and
contributes nothing to the feasible region — so it **never enters the model**; it
is **recomputed in results** from its stored coefficients × the solved
flows/sizes.

The set that must enter the model is the **transitive closure** of
`{objective} ∪ {effects with any bound}` over cross-effect chains: if a needed
effect `T` has `contribution_from = {S: …}`, then `S` must be materialized too,
because it feeds `T`. Everything outside that closure is results-only. (Getting
this closure wrong is a correctness bug, not just a perf miss — an unbounded
effect that feeds the objective must still be in the model.)

**This is already how PyPSA behaves** — and that's the strongest argument for
core. `cost` is always in the model (it is the objective); `co2` enters the model
*only* when a CO₂ `GlobalConstraint` sets a limit, and is otherwise reported
post-solve via `n.statistics`. The Effect system just makes that rule uniform and
N-ary. Consequence for the data model: **coefficients are always stored** (results
need them regardless of bounds); **only the bounded-or-objective closure is
emitted** as expressions/constraints. Effects add *no new decision variables* —
they are linear expressions of existing ones — so the continent-scale LP stays an
LP.

**Declaration vs. use — the Effect row declares; the study bounds/minimizes/
prices.** (Learning from a PyPSA prototype.) The Effect row *declares the
quantity* — unit, accounting, coefficients — study-independent and shareable with
the dataset. Everything a *study does* with that quantity lives study-side. In
particular a **limit** is a separate `GlobalConstraint` row (a new `effect_limit`
type, sibling of `primary_energy`), deliberately **not** fluxopt's
bounds-as-Effect-attributes style, for four reasons:

1. **Duals for free.** GC rows already emit `mu`; "implied carbon price" works
   with zero new code — and PyPSA-Eur reads carbon prices from exactly this `mu`.
   Bounds-on-Effect would need their own dual outputs and assignment logic.
2. **Bounds are many, effects are one.** One effect can carry a total cap, a
   per-investment-period cap, an equality, and a floor — each a sparse named GC
   row with its own sense, RHS, dual. On the Effect row that explodes into dense
   NaN-sentinel column matrices (fluxopt needs six such fields and still can't say
   "cap only 2040" without per-period arrays). Sparse rows beat dense columns.
3. **It *is* the generalization argument.** "A bound on a system-level linear
   quantity" is what GlobalConstraint has always been. `effect_limit` reads as a
   sibling of `primary_energy` (~50 lines, same dispatch/IO/duals); bounds-on-
   Effect reads as a second, parallel bounding mechanism — the framing principle
   #1 forbids.
4. **Declaration vs. use stays separated.** The Effect says *what the quantity
   is*; GC rows say *what this study does with it*. A Pareto sweep loops over GC
   rows and never touches the Effect — the workflow people already have
   (`add_co2limit` in PyPSA-Eur).

The same logic extends past bounds: **objective designation** ("which effect do I
minimize") is a study choice, so it belongs at the `optimize()` call, not as an
Effect attribute; and a **price-type cross-effect coupling** (a carbon *price*,
CO₂→cost — swept in a Pareto run exactly like a cap) is a scenario knob that
likely belongs study-side too, not baked into the `co2` effect's declaration.
Net rule: *the Effect declares the quantity; bound / minimize / price are study
verbs applied to it.*

**Caveat — per-hour rate bounds (deferred).** GlobalConstraint has no time-series
fields, so a `max_per_hour` *series* is instead a natural static-or-series Effect
attribute. Likely end state is a **hybrid**: scalar total/per-period bounds are GC
rows; time-indexed rate bounds ride the Effect — exactly how PyPSA already splits
scalar (GC) vs time-indexed (component `_t`) data. Same rule, not a contradiction.

### B. Part-load / piecewise conversion

Already wanted: PyPSA issue **#1473** (milestone v1.3, @coroa, PR #1603). Their
chosen data-model approach — "a new attribute with an extra index level for the
segments" — is exactly the device we'd reuse for effect coefficients, so the two
efforts share design language.

**What we bring:** fluxopt's `PiecewiseConversion` design as *input* to their
effort — shared interpolation weights across N flows, and auto-selection of
formulation (LP tangent constraints when convex/concave, else SOS2 /
incremental). We contribute *to their design*, we don't reopen it.

### C. Discrete single-unit multi-period investment

**What it is (fluxopt `Investment`):** decide *whether and in which period* to
build a single discrete unit, once; capacity then active for `lifetime` periods;
one-time vs recurring effect contributions.

**PyPSA gap:** multi-period investment uses `build_year` as an *input* and
continuous `p_nom`. Deciding *which* period to build a discrete unit is a MILP
extension (binary build-period variables + build-once exclusivity + lifetime
activation window).

**Native mapping:** an extendable component gains a candidate build-period set +
a binary build decision; existing `lifetime` semantics reused. Lowest priority —
flag as "phase 2, only if A and B build trust."

---

## 3. Design principles / hard constraints

Any proposal to core must honor these or it's dead on arrival:

1. **Generalize, don't parallel.** Every new concept must reduce to existing
   PyPSA behavior as a special case. Existing scripts run unchanged.
2. **Component-DataFrame native.** New data rides on component attributes with an
   extra index level (the #1473 pattern), not on side objects.
3. **linopy-native, LP stays LP by default.** No new decision variables from
   effects. MILP features (B, C) stay opt-in, as unit commitment already is.
6. **Materialize only the bounded-or-objective closure.** An effect enters the
   model iff it is the objective, carries a bound, or feeds (transitively) one
   that does. All other effects are recomputed in results from stored
   coefficients. This mirrors PyPSA's existing cost-vs-CO₂ treatment. Coefficients
   are always stored; only the closure is emitted.
4. **Single objective.** Designate *one* effect as the objective; others become
   constraints. We are **not** proposing multi-objective optimization.
5. **Backward compatibility is non-negotiable.** `marginal_cost`, `capital_cost`,
   CO₂ `GlobalConstraint` all keep working as sugar over the new machinery.
7. **Declaration vs. use.** The Effect row only *declares* the quantity. Study
   verbs — bound, minimize, price — live study-side: limits are sparse
   `effect_limit` GC rows (duals for free), the objective is an `optimize()` arg,
   prices are scenario knobs. Exception: time-indexed rate bounds (§2A caveat).

## 4. Non-goals

- Multi-objective / Pareto solving.
- Replacing the network / power-flow model (we don't touch it).
- Reproducing all of fluxopt's ergonomics — we're contributing *capabilities*,
  in PyPSA's idiom, not porting the API.

---

## 5. Strategy for landing in core

1. **Float A as a discussion, not a PR.** Open a GitHub Discussion framing the
   Effect system explicitly as "generalize the cost objective + CO₂
   GlobalConstraint into one N-quantity mechanism, existing behavior as the
   default single-effect case." Get a maintainer champion (coroa / fneum / Tom
   Brown) *before* writing code.
2. **Lead the pitch with the materialization equivalence.** The fact that PyPSA
   *already* materializes `co2` only when capped, and reports it post-solve
   otherwise, is proof the Effect rule is a generalization, not a new burden. Open
   with it.
3. **Prototype A behind the sugar.** Prove the built-in `cost` and `co2` effects
   reproduce current results bit-for-bit on an existing example, including that an
   *unbounded* `co2` effect produces identical solves (it stays out of the model)
   and only appears in results. That equivalence demo is the strongest possible
   argument.
4. **B and C are separate, later efforts.** Do not couple them to this push.

---

## 6. Open questions (for the follow-up design pass)

These need answers grounded in PyPSA's *current* source, not the abstractions
above:

1. **Exact schema.** How are effect-keyed coefficients stored on components —
   MultiIndex columns? A long-form `network.effects` + a coefficients table?
   How does this coexist with the #1473 segment index?
2. **GlobalConstraint bridge.** Can every existing `GlobalConstraint` type
   (`primary_energy`, `transmission_expansion_cost`, …) be re-expressed as an
   effect bound without behavior change? Enumerate them and check.
3. **Objective plumbing.** Where is the objective assembled in
   `pypsa/optimization/`, and how invasive is "objective = designated effect"?
4. **Cross-effect chains.** Do any existing features (carbon pricing patterns in
   PyPSA-Eur) already emulate CO₂→cost? If so, we subsume them — cite them.
5. **Periods & weighting.** How do effect totals interact with
   `investment_periods`, `objective_weightings`, and snapshot weightings?
6. **Migration & deprecation.** What's the deprecation path for `marginal_cost`
   /`capital_cost` if any — or do they stay forever as sugar (likely yes)?
7. **Maintainer appetite.** Is there an existing issue/discussion about
   generalized accounting or multiple objectives? Search before proposing.
8. **`effect_limit` GC type.** Confirm a new `effect_limit` GlobalConstraint type
   mirrors `primary_energy` dispatch/IO/duals in ~50 lines, and that per-period
   caps reuse the existing `investment_period` GC column.
9. **Declaration-vs-use boundary.** Confirmed for bounds (GC rows). Should
   objective designation and price-type cross-effect couplings also live
   study-side (`optimize()` args / GC rows) rather than on the Effect row?

---

## 7. Handoff brief (what the follow-up pass should produce)

Deliverables, in order:

1. **A grounded technical design for capability A** — read PyPSA's actual
   `components.py`, `optimization/`, `GlobalConstraint`, **and `statistics`**
   code; turn §2A and the open questions into a concrete schema +
   objective/constraint plan. Must specify: (a) how effect coefficients are
   stored on components, (b) the **materialization closure** algorithm (which
   effects get emitted vs recomputed) and where it hooks into model building,
   (c) the **results-side recomputation** path for out-of-closure effects —
   ideally reusing `n.statistics` machinery, since that's already how PyPSA
   reports uncapped emissions, (d) the backward-compat equivalence proof,
   (e) the **`effect_limit` GlobalConstraint type** (sparse limits + duals) and
   which bounds stay GC rows vs ride the Effect (per-hour caveat).
2. **A draft PyPSA GitHub Discussion post** — the "generalize what you already
   have" pitch, ready to post, sized to invite a maintainer champion.
3. **A minimal proof-of-concept plan** — the smallest change (or
   `extra_functionality` shim, as a *demonstration only*) that shows built-in
   `cost` + `co2` effects reproducing an existing PyPSA example's results.
4. **A go/no-go read** — after reading the real code, an honest call on whether A
   can land in core or is fundamentally blocked, and if blocked, exactly where.

Reference implementation to mine for semantics: fluxopt `src/fluxopt/elements.py`
(`Effect`, `Sizing`, `Investment`, `Status`, `PiecewiseConversion`) and
`src/fluxopt/model_data.py` / `constraints/`.
