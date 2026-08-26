"""Tombstone: the ``syft-client`` package has been renamed to ``syft``."""

raise ImportError(
    "syft-client has been renamed to syft (>=0.10.0). "
    "Run `pip install -U syft` and use `import syft as sy` instead of "
    "`import syft_client as sc`. Data owners running syft-bg: stop the services, "
    "then `pip install -U syft syft-bg syft-job` and re-install them. "
    "See https://github.com/OpenMined/PySyft"
)
