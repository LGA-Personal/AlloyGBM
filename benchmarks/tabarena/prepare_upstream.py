"""Install a submission-shaped payload into an otherwise unchanged pinned clone."""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
import subprocess
from pathlib import Path

UPSTREAM_REV = "041f9b3774d1191ffcdcf137cbe4e1e08159b972"
MODEL_ROOT = "packages/tabarena/src/tabarena/models/alloygbm"
EDITED = (
    "packages/tabarena/src/tabarena/models/__init__.py",
    "packages/tabarena/pyproject.toml",
    "packages/tabarena/src/tabarena/website/website_format.py",
)
HERE = Path(__file__).resolve().parent


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def registration_edits(root):
    originals = {
        p: subprocess.check_output(
            ["git", "-C", str(root), "show", f"{UPSTREAM_REV}:{p}"]
        ).decode()
        for p in EDITED
    }
    requirement = next(
        n.value.value
        for n in ast.parse((HERE / "submission/alloygbm/info.py").read_text()).body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "ALLOYGBM_REQUIREMENT"
            for t in n.targets
        )
    )
    out = dict(originals)
    out[EDITED[0]] = (
        out[EDITED[0]]
        .replace(
            "if TYPE_CHECKING:\n",
            "if TYPE_CHECKING:\n    from tabarena.models.alloygbm.model import AlloyGBMModel\n",
            1,
        )
        .replace(
            "_LAZY_CLASSES: dict[str, str] = {",
            '_LAZY_CLASSES: dict[str, str] = {\n    "AlloyGBMModel": "tabarena.models.alloygbm.model",',
            1,
        )
    )
    out[EDITED[1]] = (
        out[EDITED[1]]
        .replace(
            "[project.optional-dependencies]",
            '[project.optional-dependencies]\nalloygbm = ["' + requirement + '"]',
            1,
        )
        .replace("extended = [", 'extended = [\n  "tabarena[alloygbm]",', 1)
    )
    out[EDITED[2]] = out[EDITED[2]].replace(
        "Constants.tree: [", 'Constants.tree: [\n            "ALLOYGBM",', 1
    )
    return originals, out


def verify(root: Path):
    if git(root, "rev-parse", "HEAD") != UPSTREAM_REV:
        raise ValueError("upstream revision mismatch")
    changed = set(git(root, "diff", "--name-only", UPSTREAM_REV).splitlines())
    if changed - set(EDITED):
        raise ValueError(
            "upstream protocol/shared infrastructure changed: "
            + str(sorted(changed - set(EDITED)))
        )
    extras = set(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    payload = {
        MODEL_ROOT + "/" + p.name for p in (HERE / "submission/alloygbm").glob("*.py")
    }
    if extras - payload:
        raise ValueError(
            "upstream has unexpected untracked files: " + str(sorted(extras - payload))
        )
    _, expected = registration_edits(root)
    for name, text in expected.items():
        if (root / name).read_text() != text:
            raise ValueError("upstream registration differs: " + name)
    for source in (HERE / "submission/alloygbm").glob("*.py"):
        if (root / MODEL_ROOT / source.name).read_bytes() != source.read_bytes():
            raise ValueError("upstream payload differs: " + source.name)
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in EDITED}


def prepare(root: Path):
    root = root.resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "https://github.com/autogluon/tabarena.git",
                str(root),
            ],
            check=True,
        )
    if git(root, "rev-parse", "HEAD") != UPSTREAM_REV:
        if git(root, "status", "--porcelain"):
            raise ValueError("refusing to checkout a dirty upstream clone")
        subprocess.run(
            ["git", "-C", str(root), "checkout", "--detach", UPSTREAM_REV], check=True
        )
    originals, expected = registration_edits(root)
    for name in EDITED:
        if (root / name).read_text() not in (originals[name], expected[name]):
            raise ValueError("refusing to overwrite upstream edits: " + name)
    changed = set(git(root, "diff", "--name-only", UPSTREAM_REV).splitlines())
    if changed - set(EDITED):
        raise ValueError("upstream has changes outside the integration allowlist")
    for name, text in expected.items():
        (root / name).write_text(text)
    target = root / MODEL_ROOT
    target.mkdir(parents=True, exist_ok=True)
    for source in (HERE / "submission/alloygbm").glob("*.py"):
        destination = target / source.name
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(
                "refusing to overwrite a different model payload; remove/recreate the generated clone"
            )
        shutil.copyfile(source, destination)
    return verify(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/tabarena/upstream"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(verify(args.source) if args.check else prepare(args.source))
