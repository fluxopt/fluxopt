````# Plan: delete `docs/guide/`, consolidate into notebooks + math + API

## Context

Fluxopt currently has three parallel doc layers. The `guide/` layer is the primary source of drift because its code blocks are **not executed** at build time, while notebooks (same content, live code) are.

| Layer | Location | Executed at build | Lines |
|---|---|---|---|
| Guide | `docs/guide/*.md` | ❌ dead code | ~1150 |
| Notebooks | `docs/notebooks/*.ipynb` | ✅ yes (`mkdocs-jupyter: execute: true`) | 5 files |
| Math | `docs/math/*.md` | ❌ (pure LaTeX, no code) | ~1070 |
| API reference | autogen via `mkdocstrings` | — | — |

Decision: **delete the guide layer.** Notebooks handle "how to use X", math handles "what X means", API reference handles "what fields does X have." No parallel prose to keep in sync.

## Target state

```
docs/
├── index.md                  # lands users on notebooks/01-quickstart
├── notebooks/*.ipynb         # tutorial layer — executed on build
├── math/*.md                 # reference derivations (LaTeX)
└── api/                      # autogen from docstrings
```

`docs/guide/` is gone. `mkdocs.yml` nav has no `Guide:` section.

## Per-file migration targets

Every guide section has been classified. Destinations:

- **DUP** → already covered in a notebook or math page; delete without migrating
- **UNIQUE** → content not available elsewhere; move to the named destination
- **API-ONLY** → parameter listings that should live in docstrings (rendered via `mkdocstrings`)

### getting-started.md (122 lines)
| Section | Action |
|---|---|
| The System | DUP → nb 01, 02 |
| Step by Step (1–7) | DUP → nb 01 covers all seven steps |
| Next Steps | DELETE (stale cross-refs) |

Pure DUP — delete and redirect.

### flows.md (145 lines)
| Section | Action |
|---|---|
| Basic Construction | DUP → nb 01, 02, 03 |
| Sizing | DUP → nb 05 |
| Bounds (relative_minimum/maximum) | UNIQUE → `math/flows.md` §Bounds |
| Fixed Profiles | DUP → nb 01, 02 |
| Effect Coefficients | DUP → nb 01, 02 |
| Id Qualification | API-ONLY → `Flow` class docstring |
| Multi-Node Carriers | UNIQUE → `math/flows.md` §Multi-Node (+ nb 03 link) |
| Parameters Summary | API-ONLY → autogen from `Flow` docstring |

### converters.md (138 lines)
| Section | Action |
|---|---|
| Factory Methods (Boiler, P2H, Heat Pump, CHP) | DUP → nb 01, 02, 05 |
| Custom Conversion Factors | UNIQUE → `math/converters.md` §Custom Equations |
| Time-Varying Coefficients | UNIQUE → `math/converters.md` |
| Full Example | DUP → nb 01 |
| Parameters Summary | API-ONLY → `Converter` docstring |

### storage.md (139 lines)
| Section | Action |
|---|---|
| Basic Construction | DUP → nb 02, 05 |
| Capacity | DUP → nb 02, 05 |
| Efficiency (eta_charge/eta_discharge) | UNIQUE → `math/storage.md` §Charge Balance |
| Self-Discharge | DUP → nb 02 |
| Prior Level & Cyclic | UNIQUE → `math/storage.md` §Initial & Cyclic |
| Level Bounds | UNIQUE → `math/storage.md` §Charge State Bounds |
| Full Example | DUP → nb 02 |
| Parameters Summary | API-ONLY → `Storage` docstring |

### effects.md (219 lines) — **pilot file, do first**
| Section | Action |
|---|---|
| Defining Effects | DUP → nb 01, 05 |
| Linking Flows to Effects | DUP → nb 01, 02 |
| Bounding Effects (Total, Per-Hour) | UNIQUE → `math/effects.md` §Effect Bounds + `Effect` docstring |
| Cross-Effect Contributions (scalar, time-varying, transitive, restrictions) | UNIQUE → `math/effects.md` §Cross-Effect Contributions |
| Accessing Results | UNIQUE → `Result` class docstring with examples |
| Per-Contributor Breakdown | API-ONLY → `StatsAccessor.effect_contributions` docstring |
| Full Example | DUP → nb 01 |
| Parameters Summary | API-ONLY → `Effect` docstring |

### sizing.md (174 lines)
| Section | Action |
|---|---|
| Basic Usage | DUP → nb 05 |
| Mandatory vs Optional (incl. binary invest) | UNIQUE → `math/sizing.md` §Mandatory vs Optional + `Sizing` docstring |
| Investment Effects (Per-Size, Fixed) | UNIQUE → `math/sizing.md` §Effect Contributions |
| Storage Sizing | DUP → nb 05 |
| Interaction with Bounds | UNIQUE → `math/sizing.md` §Flow Rate Bounds with Sizing |
| Interaction with Status (big-M) | UNIQUE → `math/sizing.md` §Interaction with Status (cross-ref `status.md`) |
| Full Example | DUP → nb 05 |
| Parameters Summary | API-ONLY → `Sizing` docstring |

### status.md (221 lines)
| Section | Action |
|---|---|
| Basic Usage | DUP → nb 05 (brief) |
| Startup & Running Costs | UNIQUE → `math/status.md` §Effect Contributions + `Status` docstring |
| Duration Constraints (min/max uptime/downtime) | UNIQUE → `math/status.md` §Duration Tracking |
| Prior (Historical State) | UNIQUE → `math/status.md` §Initial Conditions + `Flow.prior_rates` docstring |
| Interaction with Sizing (big-M) | UNIQUE → `math/status.md` §Interaction with Sizing (cross-ref `sizing.md`) |
| Full Example | API-ONLY → `Status` docstring example |
| Parameters Summary | API-ONLY → `Status` + `Flow.prior_rates` docstrings |

## Phased execution

### Phase 0 — Conventions (1 PR)

1. Redirect `docs/index.md` landing → `notebooks/01-quickstart.ipynb`. Check current index.md for other links into guide.
2. Confirm `mkdocstrings` renders parameter tables acceptably from Google-style docstrings on `Flow`, `Effect`, `Storage`, `Sizing`, `Status`, `Converter`, `Investment`. If the output is flat/ugly, adjust the `mkdocstrings` options in `mkdocs.yml` **before** starting migration.
3. Add `nbval` to the dev dependency group and wire it into CI so notebooks are pytest-executed too (belt-and-braces on the claim that notebooks cover the features). Example: `uv run pytest --nbval-lax docs/notebooks/`.
4. Add `mkdocs-redirects` plugin and stage empty redirect config. Populated in Phase 1/2.
5. Add `05-investment.ipynb` to `mkdocs.yml` nav (currently on disk but not listed).

### Phase 1 — Pilot: `effects.md` (1 PR)

Execute the full migration for one file to validate the approach end-to-end. Deliverables:

- UNIQUE content merged into `docs/math/effects.md` (new §Effect Bounds, expanded §Cross-Effect Contributions with scalar/time-varying/transitive subsections)
- `Effect` class docstring updated to include full parameter table content (ensure mkdocstrings renders it usefully)
- `Result` and `StatsAccessor` docstrings updated with result-access examples
- `docs/guide/effects.md` deleted
- `mkdocs.yml` nav updated: remove `Effects: guide/effects.md` entry
- `mkdocs-redirects` entry: `guide/effects.md` → `math/effects.md`
- `grep -rn 'guide/effects' docs/ src/ README.md` — fix any cross-refs
- Docs build locally, click through nav

**Gate:** if mkdocstrings output is too poor to replace the "Parameters Summary" table, fall back to putting parameter tables in `math/*.md` (not in docstrings). This decision applies to the rest of Phase 2.

### Phase 2 — Remaining files (1 bundled PR)

Once Phase 1 settles conventions, the remaining 6 files are mechanical. Do them in a single PR to avoid review overhead:

1. `flows.md`
2. `storage.md`
3. `converters.md`
4. `sizing.md`
5. `status.md`
6. `getting-started.md` (pure delete)

Same per-file checklist as Phase 1.

### Phase 3 — Cleanup (1 PR)

- Remove `Guide:` section entirely from `mkdocs.yml`
- Sweep: `grep -rn 'docs/guide\|guide/' docs/ src/ README.md tests/` — fix any remaining links
- Verify mkdocs build has no broken links (`mkdocs build --strict`)
- Update CLAUDE.md if it references the guide layer
- Remove `docs/guide/` directory

## Per-PR checklist (applies to Phases 1–3)

- [ ] Spot-check notebook coverage claim: open the ipynb, confirm the claimed DUP section is actually demonstrated. If a notebook glosses over a detail the guide covered, **add a cell** to the notebook rather than keep the guide.
- [ ] Move UNIQUE content verbatim where possible; adapt voice only minimally (math/*.md is already reference-style; don't rewrite).
- [ ] Update docstrings with API-ONLY content.
- [ ] Delete the guide file.
- [ ] Update `mkdocs.yml` nav.
- [ ] Add `mkdocs-redirects` entry from old guide URL.
- [ ] `grep -rn 'guide/<name>' docs/ src/ README.md` and fix cross-refs.
- [ ] `mkdocs build --strict` locally — zero warnings.
- [ ] Full test suite green: `uv run pytest`, `uv run ruff check .`, `uv run mypy src/`.

## Risks & mitigations

1. **Notebook coverage gaps.** Audit classified sections as DUP based on topic scanning, not full semantic comparison. If a guide page had a nuance the notebook misses, it's lost in migration. **Mitigation:** per-PR spot-check; add notebook cells rather than keep guide content.
2. **mkdocstrings parameter tables look worse than hand-crafted markdown.** Flat output may not replace the guide's "Parameters Summary" tables gracefully. **Mitigation:** Phase 0 validates this up front. Fallback: parameter tables live in `math/*.md` instead.
3. **External bookmarks break.** Users with `/guide/effects/` bookmarks get 404s. **Mitigation:** `mkdocs-redirects` plugin. Every deleted guide URL redirects to its primary replacement.
4. **Search discoverability drops.** Guide pages index parameter names directly. **Mitigation:** math pages and docstrings already mention the same names. Verify post-migration by searching for e.g. `fixed_relative_profile` on the built site.
5. **Conventional commit scope.** This is a `docs:` change overall, but some commits will also touch `src/fluxopt/**` for docstring updates. Use `docs:` if only docs + docstrings; `refac:` if broader refactoring sneaks in.

## Out of scope

- Rewriting the math pages (only absorb guide content into them)
- Converting math/*.md into notebooks (math stays reference)
- Adding new notebooks beyond what's already on disk
- Changing the `mkdocs-material` theme or structure
- Changes to `src/fluxopt/` beyond docstring updates

## Conventions to follow

- **Commit messages**: Conventional Commits — `docs: migrate guide/effects.md to math + docstrings`
- **PR titles**: same format, ~70 char max
- **Docstring style**: Google (as configured in `mkdocs.yml`)
- **Math notation**: preserve uppercase Latin for variables (P, E, S), Greek for properties (η, δ), super/subscripts per the conventions in CLAUDE.md

## First action

Open a PR for Phase 0. Keep it small — conventions only, no content migration. Confirm mkdocstrings output quality on at least one class (`Effect`) before starting Phase 1.
````
