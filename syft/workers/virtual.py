from syft.workers.base import BaseWorker
from syft.federated import FederatedClient
from syft.workers import AbstractWorker
import syft as sy


class VirtualWorker(BaseWorker, FederatedClient):
    def _send_msg(self, message: bin, location: BaseWorker) -> bin:
        return location._recv_msg(message)

    def _recv_msg(self, message: bin) -> bin:
        return self.recv_msg(message)

    @staticmethod
    def simplify(worker: AbstractWorker) -> tuple:
        """

        """

        return (sy.serde._simplify(worker.id),)

    @staticmethod
    def detail(worker: AbstractWorker, worker_tuple: tuple) -> "VirtualWorker":
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
