"""System-level structural validation, shared by every entry path.

:func:`validate_system` is the single source of truth for "is this set of
elements a coherent system": unique ids, resolvable carrier and effect
references, and node membership. ``FlowSystem`` runs it at construction
(including the objective), and ``ModelData.build`` runs it before
materializing — so the declarative and the programmatic path reject the
same mistakes with the same messages.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel

from fluxopt.elements import PENALTY_EFFECT_ID

if TYPE_CHECKING:
    from fluxopt.components import Converter, Port
    from fluxopt.elements import Carrier, Effect, Storage


def check_unique(ids: list[str], kind: str) -> None:
    """Raise if *ids* contains duplicates.

    Args:
        ids: Identifiers to check.
        kind: Human label used in the error (e.g. ``'effect'``).
    """
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(f'Duplicate {kind} id(s): {dupes}')


def _collect_effect_refs(obj: object, out: set[str]) -> None:
    """Collect effect ids referenced by ``effects_*`` / ``contribution_from`` dicts."""
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            val = getattr(obj, name)
            if isinstance(val, dict) and (name.startswith('effects_') or name == 'contribution_from'):
                out.update(val)
            else:
                _collect_effect_refs(val, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_effect_refs(item, out)


def validate_system(
    *,
    carriers: list[Carrier],
    effects: list[Effect],
    ports: list[Port],
    converters: list[Converter],
    storages: list[Storage],
    objective: str | dict[str, float] | None = None,
) -> None:
    """Fail fast on duplicate ids and undeclared references.

    Args:
        carriers: Declared carriers.
        effects: Declared effects (the built-in penalty effect is always
            accepted as a reference target, declared or not).
        ports: Port components.
        converters: Converter components.
        storages: Storage components.
        objective: Effect name or ``{effect: weight}`` dict to validate
            against the declared effects; None skips the objective checks
            (``ModelData.build`` has no objective).
    """
    check_unique([e.id for e in effects], 'effect')
    check_unique([c.id for c in carriers], 'carrier')
    check_unique([comp.id for comp in (*ports, *converters, *storages)], 'component')

    flows = [bf for comp in (*ports, *converters, *storages) for bf in comp._qualified_flows()]
    check_unique([bf.id for bf in flows], 'flow')

    effect_ids = {e.id for e in effects} | {PENALTY_EFFECT_ID}
    refs: set[str] = set()
    for group in (effects, ports, converters, storages):
        _collect_effect_refs(group, refs)
    if unknown := sorted(refs - effect_ids):
        raise ValueError(f'Elements reference undeclared effect(s) {unknown}; declared {sorted(effect_ids)}')

    carrier_by_id = {c.id: c for c in carriers}
    if unknown := sorted({bf.flow.carrier for bf in flows} - set(carrier_by_id)):
        raise ValueError(f'Flows reference undeclared carrier(s) {unknown}; declared {sorted(carrier_by_id)}')
    for bf in flows:
        carrier = carrier_by_id[bf.flow.carrier]
        node = bf.flow.node
        if node and not carrier.nodes:
            raise ValueError(f'Flow {bf.id!r} specifies node={node!r} but carrier {carrier.id!r} has no nodes')
        if node and node not in carrier.nodes:
            raise ValueError(
                f'Flow {bf.id!r} specifies node={node!r} but carrier {carrier.id!r} only has nodes {carrier.nodes}'
            )

    if objective is not None:
        obj_keys = [objective] if isinstance(objective, str) else list(objective)
        if unknown := sorted(set(obj_keys) - effect_ids):
            raise ValueError(f'objective references undeclared effect(s) {unknown}; declared {sorted(effect_ids)}')
        if not any(k != PENALTY_EFFECT_ID for k in obj_keys):
            raise ValueError(
                'objective must name at least one non-penalty effect to minimize — '
                'the built-in penalty effect is added automatically and cannot be the sole objective'
            )
