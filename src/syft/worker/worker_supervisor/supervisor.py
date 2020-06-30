import time
import types
from syft.worker.worker_supervisor.stats import WorkerEventLog


class WorkerSupervisor(type):
    @staticmethod
    def generate_wrapper(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            name = func.__qualname__
            msg_type = None

            if name == "recv_msg":
                msg_type = type(kwargs["msg"])

            self = kwargs["self"]
            args = str(kwargs)
            event = WorkerEventLog(method_name=name, execution_time=execution_time,
                                   sizeof_object_store=len(self.object_store), args=args)
            self.worker_stats.add_event(event)
            self.worker_stats.log_msg(msg_type)

            return result
        return wrapper

    def __new__(mcs, name, bases, attrs):
        for name, target in attrs.items():
            #staticmethod check?
            if callable(target) and not isinstance(target, types.FunctionType):
                attrs[name] = WorkerSupervisor.generate_wrapper(target)
        return super(WorkerSupervisor, mcs).__new__(mcs, name, bases, attrs)
