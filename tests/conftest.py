# tests/conftest.py
"""Skip automatique des tests live si PRECITE_API_URL n'est pas defini."""
import os
import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PRECITE_API_URL"):
        return
    skip_live = pytest.mark.skip(reason="PRECITE_API_URL non defini")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)