from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
PYTHON_BINDINGS = ROOT / "bindings" / "python"

sys.path.insert(0, str(PYTHON_BINDINGS))


project = "AlloyGBM"
copyright = "2026, Logan Ashby"
author = "Logan Ashby"
release = "0.10.3"
version = "0.10.3"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_mock_imports = ["alloygbm._alloygbm"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
html_theme = "sphinx_rtd_theme"
html_title = "AlloyGBM Documentation"
html_static_path = ["_static"]
epub_show_urls = "footnote"
