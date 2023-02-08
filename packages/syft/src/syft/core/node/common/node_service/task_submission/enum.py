# stdlib
from enum import Enum


class TASK_SERVICE_FIELDS(Enum):
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    REVIEWED_BY = "reviewed_by"
    OWNER = "owner"
    USER = "user"
    CODE = "code"
    EXECUTION = "execution"
    STATUS = "status"
    UID = "uid"
    REASON = "reason"


class TASK_SERVICE_STATUS(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DENIED = "denied"


class EXECUTION_STATUS(Enum):
    WAITING = "waiting"
    ENQUEUED = "enqueued"
    RUNNING = "running"
    DONE = "done"


class TASK_SERVICE_DEFAULT_MESSAGES(Enum):
    CREATE_TASK = "Your task was successfully submited!"
    REVIEW_TASK = "Review submitted!"
