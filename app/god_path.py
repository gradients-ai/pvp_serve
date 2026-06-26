"""Locate the G.O.D checkout and put it on sys.path so `core.pvp` imports.

pvp_serve is a thin server over G.O.D's `core/pvp` game harness. Rather than
pip-installing the whole Gradients-on-Demand package (which drags in the
validator dep tree — fiber/docker/asyncpg/minio), we consume `core` straight
off a checkout: the production mechanism is a pinned git submodule at ./god,
with a sibling ../G.O.D used for local dev.

The import surface of core.pvp is self-contained (core.pvp.*, core.models.
pvp_models, core.constants + pyspiel/open_spiel/openai/numpy/pydantic/yaml), so
no validator code is pulled in.
"""

import os
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]


def _candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("GOD_REPO_PATH")
    if env:
        out.append(Path(env).expanduser().resolve())
    out.append(_REPO_ROOT / "god")  # git submodule (production)
    out.append(_REPO_ROOT.parent / "G.O.D")  # sibling checkout (local dev)
    return out


def resolve_god_path() -> Path:
    for cand in _candidates():
        if (cand / "core" / "pvp" / "bot.py").is_file():
            return cand
    tried = "\n  ".join(str(c) for c in _candidates())
    raise RuntimeError(
        "Could not locate the G.O.D checkout (need core/pvp/bot.py).\n"
        "Set GOD_REPO_PATH, add the ./god submodule, or place a sibling ../G.O.D.\n"
        f"Tried:\n  {tried}"
    )


def ensure_on_path() -> Path:
    god = resolve_god_path()
    if str(god) not in sys.path:
        sys.path.insert(0, str(god))
    return god
