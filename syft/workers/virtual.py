from time import sleep
from typing import Union

from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker


class VirtualWorker(BaseWorker):
    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        """send message to worker location"""

        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        """receive message"""

        if self.message_pending_time > 0:
            if self.verbose:
                print(f"pending time of {self.message_pending_time} seconds to receive message...")
            sleep(self.message_pending_time)

        return self.recv_msg(message)

    # For backwards compatibility with Udacity course
    @property
    def _objects(self):
        return self.object_store._objects

    @property
    def _tensors(self):
        return self.object_store._tensors

    @staticmethod
    def simplify(_worker: AbstractWorker, worker: "VirtualWorker") -> tuple:
        return BaseWorker.simplify(_worker, worker)

    @staticmethod
    def detail(worker: AbstractWorker, worker_tuple: tuple) -> Union["VirtualWorker", int, str]:
        detailed = BaseWorker.detail(worker, worker_tuple)

        if isinstance(detailed, int):
            result = VirtualWorker(id=detailed, hook=worker.hook)
        else:
            result = detailed

        return result

    @staticmethod
    def force_simplify(_worker: AbstractWorker, worker: AbstractWorker) -> tuple:
        return BaseWorker.force_simplify(_worker, worker)

    @staticmethod
    def force_detail(worker: AbstractWorker, worker_tuple: tuple) -> "VirtualWorker":
        return BaseWorker.force_detail(worker, worker_tuple)
