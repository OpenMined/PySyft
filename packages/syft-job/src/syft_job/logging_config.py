"""Shared logger setup for syft-job and the packages built on it.

``syft`` configures the ``syft`` logger only. ``syft_job``, ``syft_rds``,
``syft_enclaves`` and ``syft_bg`` are sibling top-level namespaces, so they
inherit the root logger, which has no handler. INFO records are then dropped,
and WARNING and above fall back to ``logging.lastResort`` — bare text on
stderr, with no way to raise or lower the level. Each package calls
``configure_package_logger`` once, at the end of its ``__init__``.

The helper lives here, not in ``syft``, because syft-job does not depend on
syft, while the other three packages depend on syft-job.
"""

import logging


def configure_package_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Give the ``name`` logger a handler and a level, but only if it has none.

    A caller who sets up their own logging keeps full control. Records still
    propagate to the root logger, so pytest's caplog and any root handler still
    see them; in default Python the root logger has no handler, so nothing is
    printed twice.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
