"""Print every fragment of fluxopt's math as a page of the site.

    python -m tools.program_pages

Each ``src/fluxopt/math/program/*.yaml`` is typeset by mathspec on its own and
written to ``docs/math/program/<fragment>.md``: the sets, data and variables it
declares, then every constraint and definition as the solver reads it. The
symbols come from ``docs/math/symbols.yaml``, so the pages print the notation
the hand-written math pages use. The pages are build output and are not
committed; the fragments are the source.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import mathspec
import yaml

from fluxopt.math import PROGRAM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'math' / 'program'
SYMBOLS = ROOT / 'docs' / 'math' / 'symbols.yaml'
SOURCE = 'https://github.com/fluxopt/fluxopt/blob/main/src/fluxopt/math/program/{name}.yaml'

#: The sections of a fragment that declare a name, at the top level and under ``given:``.
_SECTIONS = ('dimensions', 'relations', 'parameters', 'variables', 'expressions', 'constraints')


def _names(fragment: dict[str, object]) -> set[str]:
    """Every name a fragment declares or reads from another one."""
    names: set[str] = set()
    given = fragment.get('given') or {}
    for block in (fragment, given):
        if isinstance(block, dict):
            for section in _SECTIONS:
                names.update((block.get(section) or {}).keys())
    return names


def symbols_for(fragment: dict[str, object], table: dict[str, object]) -> dict[str, object]:
    """The symbol table cut down to the names *fragment* has, since mathspec refuses a key it cannot place."""
    names = _names(fragment)
    return {
        'notation': table['notation'],
        'dimensions': {k: v for k, v in (table.get('dimensions') or {}).items() if k in names},
        'names': {k: v for k, v in (table.get('names') or {}).items() if k in names},
    }


def _site_math(markdown: str) -> str:
    """GitHub's inline math, `` $`…`$ ``, as the ``\\(…\\)`` the site's arithmatex reads."""
    return re.sub(r'\$`(.+?)`\$', r'\\(\1\\)', markdown)


def render(path: Path, table: dict[str, object]) -> str:
    """The page for one fragment."""
    fragment = yaml.safe_load(path.read_text())
    body = mathspec.to_markdown(path, symbols=symbols_for(fragment, table))
    title = f'# {path.stem}\n\n[`{path.name}`]({SOURCE.format(name=path.stem)}), printed by mathspec.\n\n'
    return title + _site_math(body)


def main() -> int:
    """Write the page of every fragment."""
    table = yaml.safe_load(SYMBOLS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(PROGRAM.glob('*.yaml')):
        target = OUT / f'{path.stem}.md'
        target.write_text(render(path, table))
        print(f'wrote {target.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
