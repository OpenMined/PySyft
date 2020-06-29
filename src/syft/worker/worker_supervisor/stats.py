from __future__ import annotations
import syft
from dataclasses import dataclass
from collections import defaultdict
from collections import deque

LOG_QUEUE_LENGTH = 10


@dataclass
class WorkerEventLog:
    method_name: str
    execution_time: float
    sizeof_object_store: int
    args: str

    def __repr__(self):
        return f"method name: {self.method_name} - execution time: {self.execution_time} - object store size: {self.sizeof_object_store}"


@dataclass
class WorkerStats:
    message_frequency = defaultdict(int)
    event_log: deque = deque([None] * LOG_QUEUE_LENGTH, LOG_QUEUE_LENGTH)

    @syft.typecheck.type_hints
    def add_event(self, event: WorkerEventLog) -> None:
        self.event_log.appendleft(event)

    @syft.typecheck.type_hints
    def log_msg(self, msg_type: type) -> None:
        self.message_frequency[msg_type] += 1

    def __repr__(self):
        elems = []
        elems.append(f"Message stats:")
        for msg, freq in self.message_frequency.items():
            elems.append(f"\t {msg.__name__}: {freq}")

        elems.append(f"\nEvent log:")
        for log in self.event_log:
            if log:
                elems.append(f"\t {log}")
        return "\n".join(elems)
