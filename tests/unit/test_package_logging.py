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


DUPLICATE_PROBE = """
import importlib, logging
importlib.import_module({name!r})
logging.basicConfig(level=logging.INFO, format="ROOT %(message)s")
logging.getLogger({name!r} + ".child").warning("probe-record")
"""


@pytest.mark.parametrize("package", PACKAGES[1:])
def test_package_logger_does_not_duplicate_root_output(package):
    result = subprocess.run(
        [sys.executable, "-c", DUPLICATE_PROBE.format(name=package)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("probe-record") == 1, result.stderr
    assert "ROOT probe-record" in result.stderr


ENCLAVE_LEVEL_PROBE = """
import logging
from syft_enclaves.__main__ import _configure_logging
_configure_logging({level!r})
for name in ("syft_enclaves.runner", "syft_rds.client", "syft_job.job_runner"):
    log = logging.getLogger(name)
    log.debug("debug-" + name)
    log.info("info-" + name)
    log.warning("warning-" + name)
"""


@pytest.mark.parametrize(
    "level, shown, hidden",
    [
        ("WARNING", ["warning"], ["info", "debug"]),
        ("DEBUG", ["debug", "info", "warning"], []),
    ],
)
def test_enclave_log_level_applies_to_package_loggers(level, shown, hidden):
    result = subprocess.run(
        [sys.executable, "-c", ENCLAVE_LEVEL_PROBE.format(level=level)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in ("syft_enclaves.runner", "syft_rds.client", "syft_job.job_runner"):
        for kind in shown:
            assert result.stderr.count(f"{kind}-{name}") == 1, result.stderr
        for kind in hidden:
            assert f"{kind}-{name}" not in result.stderr, result.stderr
