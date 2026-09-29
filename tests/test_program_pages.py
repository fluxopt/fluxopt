"""The page each fragment of the math becomes."""

from __future__ import annotations

import pytest
import yaml

pytest.importorskip('mathspec')

from tools.program_pages import PROGRAM, SYMBOLS, _names, _site_math, render, symbols_for

TABLE = yaml.safe_load(SYMBOLS.read_text())
FRAGMENTS = sorted(PROGRAM.glob('*.yaml'))


def test_every_symbol_names_something_in_the_math():
    """The table is cut down per fragment, so a misspelled key would otherwise vanish without a word."""
    names = set().union(*(_names(yaml.safe_load(path.read_text())) for path in FRAGMENTS))
    unknown = sorted((set(TABLE['dimensions']) | set(TABLE['names'])) - names)
    assert not unknown, f'symbols.yaml names what no fragment declares: {unknown}'


def test_a_fragment_gets_only_the_symbols_it_has():
    fragment = {'dimensions': {'time': {}}, 'given': {'variables': {'rate': {}}}}
    table = symbols_for(fragment, TABLE)
    assert set(table['dimensions']) == {'time'}, 'a dimension the fragment has no use for is dropped'
    assert set(table['names']) == {'rate'}, 'a name read under given: keeps its symbol'


def test_inline_math_is_what_the_site_reads():
    assert _site_math('index $`t`$ — `time`') == r'index \(t\) — `time`', 'a code span stays a code span'


@pytest.mark.parametrize('path', FRAGMENTS, ids=[p.stem for p in FRAGMENTS])
def test_every_fragment_prints(path):
    page = render(path, TABLE)
    assert page.startswith(f'# {path.stem}\n')
    assert '$`' not in page, 'no GitHub inline math is left for the site to show as text'
