from enum import Enum
from .request_answer_message import RequestAnswerMessage
from .request_answer_response import RequestAnswerResponse


class RequestStatus(Enum):
    Pending = 1
    Rejected = 2
    Accepted = 3
