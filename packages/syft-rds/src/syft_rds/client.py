"""The Remote Data Science client.

`SyftRDSClient` COMPOSES a syft-client ``SyftboxManager`` (HAS-A, not IS-A). It
is the single place where the sync engine (``syft-client``), the dataset domain
(``syft-datasets``), the job domain (``syft-job``) are wired together.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from syft_client.sync.syftbox_manager import SyftboxManager


class SyftRDSClient:
    def __init__(self, sync_engine: SyftboxManager):
        # The nested generic sync engine (composition).
        self._sync = sync_engine

    # ------------------------------------------------------------------ #
    # nested sync engine (escape hatch for advanced/sync-only use)
    # ------------------------------------------------------------------ #
    @property
    def sync_engine(self) -> SyftboxManager:
        return self._sync

    # ------------------------------------------------------------------ #
    # delegated identity + sync surface (owned by the generic core)
    # ------------------------------------------------------------------ #
    @property
    def email(self) -> str:
        return self._sync.email

    @property
    def syftbox_folder(self) -> Path:
        return self._sync.syftbox_folder

    @property
    def has_do_role(self) -> bool:
        return self._sync.has_do_role

    @property
    def has_ds_role(self) -> bool:
        return self._sync.has_ds_role

    @property
    def peer_manager(self) -> Any:
        return self._sync.peer_manager

    def sync(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync.sync(*args, **kwargs)

    def add_peer(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync.add_peer(*args, **kwargs)

    def approve_peer_request(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync.approve_peer_request(*args, **kwargs)

    def load_peers(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync.load_peers(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # domain surface (RDS-owned)
    #
    # Currently delegated to the nested SyftboxManager; ownership migrates
    # into this class in later commits.
    # ------------------------------------------------------------------ #
    @property
    def datasets(self) -> Any:
        return self._sync.datasets

    @property
    def jobs(self) -> Any:
        return self._sync.jobs

    def submit_python_job(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync.submit_python_job(*args, **kwargs)

    def submit_bash_job(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync.submit_bash_job(*args, **kwargs)

    def process_approved_jobs(self, *args: Any, **kwargs: Any) -> Any:
        return self._sync.process_approved_jobs(*args, **kwargs)

    def __repr__(self) -> str:
        return f"SyftRDSClient(email={self._sync.email!r})"
