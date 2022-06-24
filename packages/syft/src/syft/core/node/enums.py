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
    ACCEPT = "accept"
    SOURCE = "source"
    TARGET = "target"
    NODE_ID = "node_id"
    NODE = "node"
    REQUESTED_DATE = "requested_date"
    STATUS = "status"
    NODE_NAME = "node_name"
    NODE_ADDRESS = "node_address"
    ACCEPTED_DATE = "accepted_date"
    EMAIL = "email"
    REASON = "reason"
    NAME = "name"


class AssociationRequestResponses(str, Enum):
    ACCEPT = "ACCEPTED"
    DENY = "REJECTED"
    PENDING = "PENDING"


class ResponseObjectEnum(str, Enum):
    ASSOCIATION_REQUEST = "association-request"
    DATASET = "dataset"
    GROUP = "group"
    ROLE = "role"
    USER = "user"
    WORKER = "worker"
    DATA = "data"
