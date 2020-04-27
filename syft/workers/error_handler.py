import syft
import traceback
from syft.messaging.message import StackTraceMessage
from syft.workers.base import BaseWorker
import remote_pdb


class ErrorWorkerHandler(type):
    @staticmethod
    def generate_stacktrace(e, name):
        stack = f"Exception on worker {name}:\n"
        stack = stack + "".join(traceback.TracebackException.from_exception(e).format())
        return stack

    @staticmethod
    def _wrap_send(wrapped_func):
        def wrapper_send_msg(self, message: bin, location=None) -> bin:
            try:
                return wrapped_func(self, message, location)
            except Exception as e:
                remote_pdb.set_trace()
                stack = StackTraceMessage(ErrorWorkerHandler.generate_stacktrace(e, self.id))
                bin_message = syft.serde.serialize(stack, worker=self)
                wrapped_func(self, bin_message, location)
                raise e

        return wrapper_send_msg

    @staticmethod
    def _wrap_recv(wrapped_func):
        def wrapper_recv_msg(self, message: bin) -> bin:
            try:
                return wrapped_func(self, message)
            except Exception as e:
                stack = StackTraceMessage(ErrorWorkerHandler.generate_stacktrace(e, self.id))
                bin_message = syft.serde.serialize(stack, worker=self)
                wrapped_func(self, bin_message)
                raise e

        return wrapper_recv_msg

    def __new__(meta, classname, bases, class_dict):
        # if "_send_msg" in class_dict:
        #     class_dict["_send_msg"] = ErrorWorkerHandler._wrap_send(class_dict["_send_msg"])

        if "_recv_msg" in class_dict:
            class_dict["_recv_msg"] = ErrorWorkerHandler._wrap_recv(class_dict["_recv_msg"])

        return super(ErrorWorkerHandler, meta).__new__(meta, classname, bases, class_dict)


class VirtualErrorWorkerHandler(type(BaseWorker), ErrorWorkerHandler):
    pass


class WebsocketClientErrorHandler(type(BaseWorker), ErrorWorkerHandler):
    pass
