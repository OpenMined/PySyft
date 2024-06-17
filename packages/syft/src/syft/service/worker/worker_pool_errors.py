# relative
from ...types.errors import SyftException


class WorkerPoolInvalidError(SyftException):
    public_message = (
        "You tried to run a syft function attached to a worker pool in blocking mode,"
        " which is currently not supported. Run your function with `blocking=False` to"
        " run as a job on your worker pool"
    )
