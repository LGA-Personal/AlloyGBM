"""Python runtime integration checks for the native alloygbm extension."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NativeRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._repo_root = Path(__file__).resolve().parents[3]
        cls._tempdir = tempfile.TemporaryDirectory(prefix="alloygbm-runtime-")
        temp_root = Path(cls._tempdir.name)
        wheel_out = temp_root / "wheelhouse"
        install_target = temp_root / "site-packages"
        wheel_out.mkdir(parents=True, exist_ok=True)
        install_target.mkdir(parents=True, exist_ok=True)

        common_env = dict(os.environ)
        common_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--manifest-path",
                str(cls._repo_root / "bindings/python/Cargo.toml"),
                "--interpreter",
                sys.executable,
                "--out",
                str(wheel_out),
                "-q",
            ],
            cwd=cls._repo_root,
            env=common_env,
            check=True,
        )

        wheels = sorted(wheel_out.glob("alloygbm-*.whl"))
        if not wheels:
            raise RuntimeError("maturin build did not produce an alloygbm wheel")
        wheel = wheels[-1]

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-cache-dir",
                "--target",
                str(install_target),
                str(wheel),
            ],
            cwd=cls._repo_root,
            env=common_env,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        cls._install_path = str(install_target)
        sys.path.insert(0, cls._install_path)

        for module_name in ("alloygbm", "alloygbm._alloygbm", "alloygbm.regressor"):
            sys.modules.pop(module_name, None)

        import alloygbm

        cls.alloygbm = alloygbm

    @classmethod
    def tearDownClass(cls) -> None:
        for module_name in ("alloygbm", "alloygbm._alloygbm", "alloygbm.regressor"):
            sys.modules.pop(module_name, None)

        if hasattr(cls, "_install_path") and cls._install_path in sys.path:
            sys.path.remove(cls._install_path)

        if hasattr(cls, "_tempdir"):
            cls._tempdir.cleanup()

    def test_runtime_import_exposes_native_runtime_info(self) -> None:
        info = self.alloygbm.native_runtime_info()
        self.assertEqual(info.name, "alloygbm")
        self.assertRegex(info.version, re.compile(r"^\d+\.\d+\.\d+"))

    def test_runtime_native_predictor_entrypoint_executes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "serialization|artifact|header"):
            self.alloygbm._alloygbm.predictor_predict_batch(
                b"invalid-artifact", [[1.0, 0.0]]
            )

    def test_public_regressor_bridge_uses_native_extension_runtime(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "serialization|artifact|header"):
            self.alloygbm.GBMRegressor.predict_from_artifact(
                b"invalid-artifact", [[1.0, 0.0]]
            )


if __name__ == "__main__":
    unittest.main()
