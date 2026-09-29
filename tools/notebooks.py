"""Execute the tutorial notebooks and write each one as the page the site builds.

    python -m tools.notebooks

zensical builds Markdown and has no notebook plugin, so every
``docs/notebooks/**/*.ipynb`` is run top to bottom and written beside itself as
``.md``: a markdown cell as it is, a code cell as a ``python`` fence, and each
output as what a browser needs to show it. The ``.md`` files are build output
and are not committed; the notebook stays the source a reader downloads.

A cell that raises fails the command, so a notebook the API has left behind
fails the docs build.
"""

from __future__ import annotations

from pathlib import Path

import nbclient
import nbformat

ROOT = Path(__file__).resolve().parent.parent / 'docs' / 'notebooks'


def _fence(text: str, lang: str) -> str:
    """A fence long enough that no backtick run inside *text* closes it."""
    run = 3
    while '`' * run in text:
        run += 1
    return f'{"`" * run}{lang}\n{text.rstrip()}\n{"`" * run}'


def _output(output: nbformat.NotebookNode) -> str | None:
    """One output as Markdown, preferring what renders over what reads."""
    if output.output_type == 'stream':
        return _fence(output.text, 'text')
    if output.output_type == 'error':
        return _fence('\n'.join(output.traceback), 'text')
    data = output.get('data', {})
    if 'text/html' in data:
        return f'<div markdown="0">\n{data["text/html"].strip()}\n</div>'
    if 'image/svg+xml' in data:
        return data['image/svg+xml']
    for mime in ('image/png', 'image/jpeg'):
        if mime in data:
            return f'<img src="data:{mime};base64,{data[mime].strip()}" alt="">'
    if 'text/markdown' in data:
        return data['text/markdown']
    if 'text/plain' in data:
        return _fence(data['text/plain'], 'text')
    return None


def render(path: Path) -> str:
    """Execute the notebook at *path* and return it as a Markdown page.

    Raises:
        nbclient.exceptions.CellExecutionError: If a cell raises.
    """
    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, resources={'metadata': {'path': str(path.parent)}}).execute()
    blocks: list[str] = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            blocks.append(cell.source)
            if len(blocks) == 1:
                blocks.append(f'[Download this notebook]({path.name})')
        elif cell.cell_type == 'code' and cell.source.strip():
            blocks.append(_fence(cell.source, 'python'))
            blocks.extend(rendered for output in cell.outputs if (rendered := _output(output)) is not None)
    return '\n\n'.join(blocks) + '\n'


def main() -> int:
    """Write the page of every notebook under ``docs/notebooks``."""
    for path in sorted(ROOT.rglob('*.ipynb')):
        if '.ipynb_checkpoints' in path.parts:
            continue
        path.with_suffix('.md').write_text(render(path))
        print(f'wrote {path.with_suffix(".md").relative_to(ROOT.parent.parent)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
