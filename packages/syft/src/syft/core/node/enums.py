# stdlib
from enum import Enum

# relative
from ...util import bcolors


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


class ResponseObjectEnum(str, Enum):
    ASSOCIATION_REQUEST = "association-request"
    DATASET = "dataset"
    GROUP = "group"
    ROLE = "role"
    USER = "user"
    WORKER = "worker"
    DATA = "data"


class PointerStatus(str, Enum):
    READY = bcolors.green("Ready")
    PROCESSING = bcolors.yellow("Processing")
