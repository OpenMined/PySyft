# stdlib
from typing import Union

# third party
from google.protobuf.message import Message
from typing_extensions import TypeAlias

Deserializeable: TypeAlias = Union[str, dict, bytes, Message]
