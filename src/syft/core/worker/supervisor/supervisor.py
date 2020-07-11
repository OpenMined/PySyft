import time
from typing import Callable

from ....typecheck import type_hints
from .stats import WorkerEventLog


class WorkerSupervisor(type):
    @staticmethod
    @type_hints
    def generate_wrapper(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            name = func.__qualname__
            msg_type = None

            if func.__name__ == "recv_msg":
                msg_type = type(kwargs["msg"])

            self = args[0]
            args = str(kwargs)
            event = WorkerEventLog(
                method_name=name,
                start_time=start_time,
                execution_time=execution_time,
                sizeof_object_store=0,
                # len(self.object_store)
                args=args,
            )
            self.worker_stats.add_event(event=event)
            if msg_type:
                self.worker_stats.log_msg(msg_type=msg_type)

            return result

        return wrapper

    def __new__(mcs, name, bases, attrs):
        for name, target in attrs.items():
            if callable(target) and not isinstance(target, staticmethod):
                attrs[name] = WorkerSupervisor.generate_wrapper(func=target)
        return super(WorkerSupervisor, mcs).__new__(mcs, name, bases, attrs)
