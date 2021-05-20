# stdlib
from enum import Enum


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
    VALUE = "value"
    HANDSHAKE = "handshake"
    SENDER_ADDRESS = "sender_address"
    DOMAIN_ADDRESS = "domain_address"


class AssociationRequestResponses(str, Enum):
    ACCEPT = "accept"
    DENY = "deny"


class ResponseObjectEnum(str, Enum):
    ASSOCIATION_REQUEST = "association-request"
    DATASET = "dataset"
    GROUP = "group"
    ROLE = "role"
    USER = "user"
    WORKER = "worker"
    DATA = "data"
