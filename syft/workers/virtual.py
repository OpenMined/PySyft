from syft.workers.base import BaseWorker
from syft.federated import FederatedClient
from syft.workers import AbstractWorker
import syft as sy
from syft.generic import pointers


class VirtualWorker(BaseWorker, FederatedClient):
    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        return self.recv_msg(message)

    @staticmethod
    def force_simplify(worker: AbstractWorker) -> tuple:
        return (sy.serde._simplify(worker.id), sy.serde._simplify(worker._objects), worker.auto_add)

    @staticmethod
    def force_detail(worker: AbstractWorker, worker_tuple: tuple) -> tuple:
        worker_id, _objects, auto_add = worker_tuple
        worker_id = sy.serde._detail(worker, worker_id)

        result = sy.VirtualWorker(sy.hook, worker_id, auto_add=auto_add)
        _objects = sy.serde._detail(worker, _objects)
        result._objects = _objects

        # make sure they weren't accidentally double registered
        for _, obj in _objects.items():
            if obj.id in worker._objects:
                del worker._objects[obj.id]

        return result

    @staticmethod
    def simplify(worker: AbstractWorker) -> tuple:
        return (sy.serde._simplify(worker.id),)

    @staticmethod
    def detail(worker: AbstractWorker, worker_tuple: tuple) -> "pointers.PointerTensor":
        """
        This function reconstructs a PlanPointer given it's attributes in form of a tuple.

        Args:
            worker: the worker doing the deserialization
            plan_pointer_tuple: a tuple holding the attributes of the PlanPointer
        Returns:
            PointerTensor: a PointerTensor
        Examples:
            ptr = detail(data)
        """
        worker_id = sy.serde._detail(worker, worker_tuple[0])

        referenced_worker = worker.get_worker(worker_id)

        return referenced_worker
