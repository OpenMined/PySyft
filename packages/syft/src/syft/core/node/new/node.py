# stdlib
from typing import Optional

# third party
from nacl.signing import SigningKey

# relative
from ...common.uid import UID


class NewNode:
    id: Optional[UID]
    name: Optional[str]
    signing_key: Optional[SigningKey]
