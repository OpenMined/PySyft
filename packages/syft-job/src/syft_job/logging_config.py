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


PACKAGE_LOGGERS = ("syft_job", "syft_rds", "syft_enclaves", "syft_bg")


class _FallbackHandler(logging.StreamHandler):
    """Prints a record only while the root logger has no handler.

    When a program configures the root logger, records propagate to it and
    print there, so this handler stays silent and nothing prints twice.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if not logging.getLogger().handlers:
            super().emit(record)


def configure_package_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Give the ``name`` logger a fallback handler and a level, if it has none.

    Records still propagate to the root logger, so pytest's caplog and any
    root handler see them. A program that configures the root logger calls
    ``defer_to_root_logging`` to make the root level apply here too.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _FallbackHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def defer_to_root_logging(names: tuple[str, ...] = PACKAGE_LOGGERS) -> None:
    """Remove the level that ``configure_package_logger`` set on each logger.

    Call it after ``logging.basicConfig``. The root level then applies to
    these loggers, as it does to every logger with no level of its own.
    """
    for name in names:
        logger = logging.getLogger(name)
        if any(isinstance(h, _FallbackHandler) for h in logger.handlers):
            logger.setLevel(logging.NOTSET)
