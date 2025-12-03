from __future__ import annotations

import os

def default_import_root() -> str:
    # Prefer an explicit env from devcontainer.json
    env = os.environ.get("APP_DEFAULT_IMPORT_DIR")
    if env and os.path.isdir(env):
        return env
        # Common bind targets
    for cand in ("/hosthome", "/hosthome_win", "/imports"):
        if os.path.isdir(cand):
            return cand
        # fallback to the container user's home
    home = os.path.expanduser("~")
    return home if os.path.isdir(home) else "/"