import syft
import traceback
from syft.messaging.message import StackTraceMessage
from syft.workers.base import BaseWorker


class ErrorWorkerHandler(type):
    def error_cleanup(cls):
        raise NotImplementedError

    def error_worker_info(cls):
        raise NotImplementedError

    @staticmethod
    def _wrap_send(wrapped_func):
        def wrapper_send_msg(self, message: bin, location=None) -> bin:
            try:
                return wrapped_func(self, message, location)
            except Exception as e:
                stack_string = self.error_worker_info() + "".join(traceback.TracebackException.from_exception(e).format())
                stack = StackTraceMessage(stack_string)
                bin_message = syft.serde.serialize(stack, worker=self)
                self.ws.send(bin_message)
                self.error_cleanup()
                raise e

        return wrapper_send_msg

    @staticmethod
    def _wrap_recv(wrapped_func):
        def wrapper_recv_msg(self, message: bin) -> bin:
            try:
                return wrapped_func(self, message)
            except Exception as e:
                # stack_string = self.error_worker_info() + "".join(traceback.TracebackException.from_exception(e).format())
                # stack = StackTraceMessage(stack_string)
                # bin_message = syft.serde.serialize(stack, worker=self)
                self.ws.send("Test")
                self.error_cleanup()
                raise e

        return wrapper_recv_msg

    def __new__(meta, classname, bases, class_dict):
        # if "_send_msg" in class_dict:
        #     class_dict["_send_msg"] = ErrorWorkerHandler._wrap_send(class_dict["_send_msg"])

        if "_recv_msg" in class_dict:
            class_dict["_recv_msg"] = ErrorWorkerHandler._wrap_recv(class_dict["_recv_msg"])

        class_dict["error_cleanup"] = meta.error_cleanup
        class_dict["error_worker_info"] = meta.error_worker_info
        return super(ErrorWorkerHandler, meta).__new__(meta, classname, bases, class_dict)


class VirtualErrorWorkerHandler(type(BaseWorker), ErrorWorkerHandler):
    def error_cleanup(self):
        pass

    def error_worker_info(self):
        return f"VirtualWorker {self.id} raised the following error:\n"

class WebsocketClientErrorHandler(type(BaseWorker), ErrorWorkerHandler):
    def error_cleanup(self):
        self.ws.close()

    def error_worker_info(self):
        return f"WebsockerClientWorker {self.id}, host {self.host}, port {self.port} raised the following error:\n"
