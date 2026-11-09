"""Static checks that every notebook is valid and wired to the project modules."""

from pathlib import Path

import nbformat
import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def test_all_four_notebooks_present():
    assert [p.name for p in NOTEBOOKS] == [
        "01_preprocess.ipynb",
        "02_build_vectorstore.ipynb",
        "03_rag_queries.ipynb",
        "04_analysis.ipynb",
    ]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_non_empty(path):
    """Regression: two notebooks were once committed as 0-byte files."""
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)

    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(code) >= 3
    assert any("from src." in src for src in code)
    for src in code:
        compile(src, f"{path.name} cell", "exec")
