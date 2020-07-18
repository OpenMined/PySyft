from ..abstract.service import WorkerService  # noqa: F401
from typing import Dict  # noqa: F401

message_service_mapping: Dict[str, WorkerService] = {}

from . import service  # noqa: F401
