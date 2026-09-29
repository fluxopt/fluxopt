"""The page each tutorial notebook becomes, output by output."""

from __future__ import annotations

import pytest

nbformat = pytest.importorskip('nbformat')
pytest.importorskip('nbclient')

from tools.notebooks import _fence, _output  # noqa: E402


def test_a_fence_outlasts_every_backtick_run_inside_it():
    fenced = _fence('```python\nx = 1\n```', 'markdown')
    assert fenced.startswith('````markdown\n'), 'a three-backtick run inside needs a four-backtick fence'
    assert fenced.endswith('\n````')


@pytest.mark.parametrize(
    ('output', 'expected'),
    [
        pytest.param(
            nbformat.v4.new_output('stream', name='stdout', text='Total cost: 1.00\n'),
            '```text\nTotal cost: 1.00\n```',
            id='printed-text',
        ),
        pytest.param(
            nbformat.v4.new_output('display_data', data={'text/html': '<div id="fig"></div>', 'text/plain': 'Figure'}),
            '<div markdown="0">\n<div id="fig"></div>\n</div>',
            id='html-over-plain-text',
        ),
        pytest.param(
            nbformat.v4.new_output('display_data', data={'image/png': 'iVBORw0KGgo=\n'}),
            '<img src="data:image/png;base64,iVBORw0KGgo=" alt="">',
            id='an-image-inline',
        ),
        pytest.param(
            nbformat.v4.new_output('execute_result', data={'text/plain': '<xarray.DataArray>'}, execution_count=1),
            '```text\n<xarray.DataArray>\n```',
            id='a-repr',
        ),
    ],
)
def test_an_output_renders_as_what_a_browser_shows(output, expected):
    assert _output(output) == expected


def test_an_output_with_nothing_a_page_can_show_is_dropped():
    output = nbformat.v4.new_output('display_data', data={'application/vnd.custom+json': {}})
    assert _output(output) is None
