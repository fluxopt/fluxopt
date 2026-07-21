"""JSON Schema generation for the element layer.

The user-facing elements are ``pydantic.dataclasses``, so pydantic can emit a
JSON Schema for each type. This is the machine-readable contract behind a future
config front-end, GUI, or LLM-assisted authoring surface.

Array-valued ``Variate`` fields (inline time-series) appear as permissive
``{}`` in the schema — profiles are meant to live in data files and be
referenced, not inlined (see ``docs/design/config-and-pydantic-direction.md``).
Full instance round-trip (``to_dict``/``from_dict``) is a separate follow-up: it
needs ``IdList`` serialization and idempotent component qualification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from fluxopt.components import Converter, Port
from fluxopt.elements import Carrier, Effect, Flow, Investment, Sizing, Status, Storage

if TYPE_CHECKING:
    from collections.abc import Mapping

ELEMENT_TYPES: tuple[type, ...] = (
    Carrier,
    Effect,
    Flow,
    Sizing,
    Investment,
    Status,
    Storage,
    Port,
    Converter,
)
"""Every user-facing element type, in declaration order."""


def element_schema(element_type: type) -> dict[str, Any]:
    """Return the JSON Schema for one element type.

    Args:
        element_type: An element dataclass (e.g. ``Flow``, ``Effect``).
    """
    return TypeAdapter(element_type).json_schema()


def all_element_schemas() -> Mapping[str, dict[str, Any]]:
    """Return JSON Schemas for every element type, keyed by class name."""
    return {t.__name__: element_schema(t) for t in ELEMENT_TYPES}
