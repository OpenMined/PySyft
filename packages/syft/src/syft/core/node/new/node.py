# stdlib
from typing import Optional

# relative
from ...common.uid import UID
from .credentials import SyftSigningKey


class NewNode:
    id: Optional[UID]
    name: Optional[str]
    signing_key: Optional[SyftSigningKey]
