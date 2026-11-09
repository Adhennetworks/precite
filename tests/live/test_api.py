import os
import pytest

API_URL = os.environ.get("PRECITE_API_URL", "").rstrip("/")
if not API_URL:
    pytest.skip("PRECITE_API_URL non defini", allow_module_level=True)

pytestmark = pytest.mark.live