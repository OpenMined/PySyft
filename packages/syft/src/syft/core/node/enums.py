# stdlib
from enum import Enum
import functools

# relative
from ..common.serde import recursive_serde_register
from ..common.serde.recursive_primitives import deserialize_enum
from ..common.serde.recursive_primitives import serialize_enum


class PyGridClientEnums(str, Enum):
    ENCODING = "ISO-8859-1"


class RequestAPIFields(str, Enum):
    ADDRESS = "address"
    UID = "uid"
    POINTABLE = "pointable"
    QUERY = "query"
    CONTENT = "content"
    REPLY_TO = "reply_to"
    MESSAGE = "message"
    ERROR = "error"
    RESPONSE = "response"
    SOURCE = "source"
    TARGET = "target"
    NODE = "node"
    REQUESTED_DATE = "requested_date"
    STATUS = "status"


class AssociationRequestResponses(str, Enum):
    ACCEPT = "ACCEPTED"
    DENY = "REJECTED"
    PENDING = "PENDING"


recursive_serde_register(
    AssociationRequestResponses,
    serialize=serialize_enum,
    deserialize=functools.partial(deserialize_enum, AssociationRequestResponses),
)


class ResponseObjectEnum(str, Enum):
    ASSOCIATION_REQUEST = "association-request"
    DATASET = "dataset"
    GROUP = "group"
    ROLE = "role"
    USER = "user"
    WORKER = "worker"
    DATA = "data"
