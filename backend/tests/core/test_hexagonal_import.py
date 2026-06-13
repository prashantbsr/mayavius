"""T-130 — the hexagonal import test (spec/10 §1.2).

Import every module under ``app.core.*`` in a clean subprocess (so import side
effects don't leak from other tests), then assert ``sys.modules`` contains NO key
whose top package is in {fastapi, starlette, torch, uvicorn}, no key starting with
``app.adapters``, and no numpy ``cuda`` key. This is the single guardrail behind the
hexagonal mandate (handover §3 / spec/03) — it fails CI if the core's dependency
edge is ever violated.
"""

from __future__ import annotations

import subprocess
import sys

# The exact one-liner from spec/10 §1.2. Walk every app.core submodule, then collect
# any banned module that ended up imported, and assert the set is empty.
_ONE_LINER = (
    "import importlib, pkgutil, app.core, sys; "
    "[importlib.import_module(m.name) "
    "for m in pkgutil.walk_packages(app.core.__path__, 'app.core.')]; "
    "banned={k for k in sys.modules "
    "if k.split('.')[0] in {'fastapi','starlette','torch','uvicorn'} "
    "or k.startswith('app.adapters') "
    "or ('cuda' in k and k.split('.')[0]=='numpy')}; "
    "assert not banned, banned"
)


def test_core_imports_no_framework() -> None:
