"""Shared pytest config for Data/ tests.

Adds `Data/` (so `pipeline.*` imports work) and `Data/WebDownloads/`
(so `crawl_fit_pdfs` and `llm_ocr_pdf` modules import) to sys.path.

Stubs heavy optional dependencies (sentence_transformers, torch) that are
only needed at runtime for embedding/OCR — unit tests do not exercise them.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent
_WEB_DIR = _DATA_DIR / "WebDownloads"

for path in (_DATA_DIR, _WEB_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# --- Stub heavy optional deps so `pipeline/__init__.py` import chain works.
def _install_stub(module_name: str, **attrs) -> None:
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)
    for attr_name, attr_val in attrs.items():
        setattr(stub, attr_name, attr_val)
    sys.modules[module_name] = stub


class _SentenceTransformerStub:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("SentenceTransformer is stubbed in unit tests")


_install_stub("sentence_transformers", SentenceTransformer=_SentenceTransformerStub)
