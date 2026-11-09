"""Configuration pytest globale.

Regles :
- Les tests `live` sont skip automatiquement si PRECITE_API_URL n'est pas defini.
- Les tests `quality` sont exclus par defaut (voir addopts dans pyproject.toml).
- Les tests `integration` sont exclus en CI rapide si besoin (via -m).
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip les tests live si l'URL de l'API n'est pas fournie.

    Note : on marque skip a l'execution, pas a la collection, pour que les
    tests restent visibles dans `pytest --collect-only` (utile pour verifier
    qu'on ne perd pas de couverture).
    """
    if os.environ.get("PRECITE_API_URL"):
        return

    skip_live = pytest.mark.skip(reason="PRECITE_API_URL non defini")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)