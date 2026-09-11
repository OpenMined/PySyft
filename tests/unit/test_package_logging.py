"""Each package must configure its own logger when imported on its own.

`syft/__init__.py` configures only the `syft` logger. The other packages are
sibling top-level namespaces, so without this they inherit the root logger —
no handler, level WARNING — and every logger.info/warning/error call in them
is dropped, swallowed tracebacks included.

Each case runs in its own interpreter: the point is that importing the package
alone is enough, with no dependence on `syft` being imported first.
"""

import subprocess
import sys

import pytest

PACKAGES = ["syft", "syft_job", "syft_rds", "syft_enclaves", "syft_bg"]

PROBE = """
import importlib, logging, sys
importlib.import_module({name!r})
logger = logging.getLogger({name!r})
print(logger.getEffectiveLevel(), bool(logger.handlers))
"""


@pytest.mark.parametrize("package", PACKAGES)
def test_package_logger_is_audible(package):
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(name=package)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    level, has_handler = result.stdout.strip().splitlines()[-1].split()
    assert int(level) == 20, f"{package} effective level is {level}, want INFO (20)"
    assert has_handler == "True", f"{package} logger has no handler"
