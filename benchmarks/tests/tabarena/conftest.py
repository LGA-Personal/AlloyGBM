"""Optional integration suite: run in the pinned, isolated benchmark environment."""

import pytest

pytest.importorskip("tabarena", reason="use the isolated TabArena environment")
