from enum import Enum
from .request_answer_message import RequestAnswerMessage, RequestAnswerMessageService
from .request_answer_response import RequestAnswerResponse, RequestAnswerResponseService
from .request_answer import RequestService, RequestMessage


class RequestStatus(Enum):
    Pending = 1
    Rejected = 2
    Accepted = 3
