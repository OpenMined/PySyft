"""client.api: every api the current user can read, across datasites."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from syft_rds.apis.api import Api
from syft_rds.apis.models import API_FILE_NAME, APIS_DIR, load_api_definition
from syft_rds.apis.repr import api_collection_repr_html, api_collection_repr_str

if TYPE_CHECKING:
    from syft_rds.client import SyftRDSClient

logger = logging.getLogger(__name__)


class ApiCollection:
    """Apis synced into this SyftBox. Access one with client.api.<name>."""

    def __init__(self, client: SyftRDSClient):
        self._client = client

    def get_all(self) -> list[Api]:
        apis = []
        for api_dir in self._api_dirs():
            try:
                definition = load_api_definition(api_dir)
            except Exception as e:
                logger.warning(f"Skipping unreadable api at {api_dir}: {e}")
                continue
            datasite = api_dir.parents[2].name
            apis.append(Api(api_dir.name, datasite, api_dir, definition, self._client))
        return apis

    def _api_dirs(self) -> list[Path]:
        syftbox_folder = Path(self._client.syftbox_folder)
        pattern = f"*@*/{APIS_DIR.as_posix()}/*/{API_FILE_NAME}"
        return sorted(p.parent for p in syftbox_folder.glob(pattern))

    def get(self, name: str, datasite: str | None = None) -> Api:
        matches = [
            api
            for api in self.get_all()
            if api.name == name and datasite in (None, api.datasite)
        ]
        if not matches:
            raise KeyError(f"No api named '{name}'. Available: {self._names()}")
        if len(matches) > 1:
            owners = [api.datasite for api in matches]
            raise KeyError(
                f"Api '{name}' exists on several datasites {owners}; use "
                f"client.api.get('{name}', datasite=...)"
            )
        return matches[0]

    def _names(self) -> list[str]:
        return sorted({api.name for api in self.get_all()})

    def __getitem__(self, key: str | int) -> Api:
        if isinstance(key, int):
            return self.get_all()[key]
        return self.get(key)

    def __getattr__(self, name: str) -> Api:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.get(name)
        except KeyError as e:
            raise AttributeError(str(e)) from None

    def __dir__(self) -> list[str]:
        names = [n for n in self._names() if n.isidentifier()]
        return sorted(set(super().__dir__()) | set(names))

    def __iter__(self) -> Iterator[Api]:
        return iter(self.get_all())

    def __len__(self) -> int:
        return len(self.get_all())

    def __repr__(self) -> str:
        return api_collection_repr_str(self.get_all())

    def _repr_html_(self) -> str:
        return api_collection_repr_html(self.get_all())
