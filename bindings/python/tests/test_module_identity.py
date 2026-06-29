"""Regression tests for GBMRegressor's public module identity (issue #48).

After the v0.12.3 ``_regressor/`` package refactor, ``GBMRegressor`` is
defined inside ``alloygbm._regressor._core`` but its public, stable import
path is ``alloygbm.regressor`` (a back-compat shim that re-exports the
class). Without explicitly setting ``__module__``, pickle payloads and
``repr`` leak the private implementation path, tying the pickle format to
the internal package layout. These tests pin the public identity so a
future refactor of ``_regressor``'s internals cannot break existing pickles
or surface the private path through introspection.
"""

import pickle

import alloygbm
from alloygbm import GBMRegressor


def test_package_exposes_version():
    assert alloygbm.__version__ == "0.12.9"


def test_gbmregressor_module_is_public_shim_path():
    """``GBMRegressor.__module__`` must advertise the public shim path."""
    assert GBMRegressor.__module__ == "alloygbm.regressor"


def test_gbmregressor_pickle_uses_public_module_path():
    """A newly-pickled ``GBMRegressor`` instance must reference its class
    via the public shim path so that ``_regressor``'s internal layout can
    evolve without invalidating pickles."""
    payload = pickle.dumps(GBMRegressor())
    assert b"alloygbm.regressor" in payload
    assert b"_regressor._core" not in payload
