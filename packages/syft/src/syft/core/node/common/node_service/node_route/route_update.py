# future
from __future__ import annotations

# stdlib
from typing import Any
from typing import KeysView
from typing import Optional

# third party
from pydantic import BaseModel


class RouteUpdate(BaseModel):
    source_node_uid: str
    source_node_url: Optional[str]
    private: bool = False
    autodetect: bool = False

    # allows splatting with **payload
    def keys(self) -> KeysView[str]:
        return self.__dict__.keys()

    # allows splatting with **payload
    def __getitem__(self, key: str) -> Any:
        return self.__dict__.__getitem__(key)

    def validate(self) -> bool:
        if self.autodetect:
            return True
        elif self.source_node_url:
            return True
        raise Exception("source_node_url must be supplied if autodetect=False")
