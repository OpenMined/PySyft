"""Importing optional dependencies with an actionable error.

Verifying Tinfoil evidence needs the ``tinfoil`` SDK, which most installs do
not want: it pulls in ``openai``, ``sigstore`` and ``pyopenssl``. So it lives
behind an extra, and every import of it goes through :func:`require` so a
missing install produces install instructions rather than a bare ImportError.
"""

from __future__ import annotations

import importlib
from types import ModuleType

PACKAGE = "syft-enclave"


class MissingOptionalDependency(ImportError):
    """An optional dependency is needed for the requested feature.

    Subclasses ``ImportError`` so existing ``except ImportError`` handlers keep
    working.
    """


def require(module: str, *, extra: str, feature: str, docs: str = "") -> ModuleType:
    """Import *module*, or raise with how to install it.

    Args:
        module: the module to import, e.g. ``"tinfoil.attestation"``.
        extra: the extra that provides it, e.g. ``"tinfoil"``.
        feature: what the caller was trying to do, for the message.
        docs: optional repo-relative doc path to point at.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingOptionalDependency(
            _message(module, extra=extra, feature=feature, docs=docs)
        ) from e


def _message(module: str, *, extra: str, feature: str, docs: str) -> str:
    root = module.split(".")[0]
    lines = [
        f"{feature} requires the optional '{root}' package, which is not installed.",
        "",
        "Install it with one of:",
        "",
        f'    pip install "{PACKAGE}[{extra}]"',
        f'    uv pip install "{PACKAGE}[{extra}]"',
        "",
        "or install the package on its own:",
        "",
        f"    pip install {root}",
    ]
    if docs:
        lines += ["", f"See {docs} for the full deploy and verify flow."]
    return "\n".join(lines)
