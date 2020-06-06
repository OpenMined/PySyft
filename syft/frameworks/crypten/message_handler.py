import types

from syft.messaging.message import CryptenInitPlan
from syft.messaging.message import CryptenInitJail
from syft.messaging.message import ObjectMessage

from syft.frameworks.crypten.context import run_party

from syft.frameworks.crypten.jail import JailRunner
from syft.frameworks.crypten import utils
from syft.workers.base import BaseWorker

from syft.generic.abstract.message_handler import AbstractMessageHandler


def get_worker_from_rank(worker: BaseWorker, rank: int) -> BaseWorker:
    assert hasattr(worker, "rank_to_worker_id"), "First need to call run_crypten_party"
    return worker._get_worker_based_on_id(worker.rank_to_worker_id[rank])


class CryptenMessageHandler(AbstractMessageHandler):
    def __init__(self, object_store, worker):
        super().__init__(object_store)
        self.worker = worker
        setattr(worker, "get_worker_from_rank", types.MethodType(get_worker_from_rank, worker))

    def init_routing_table(self):
        return {
            CryptenInitPlan: self.run_crypten_party_plan,
            CryptenInitJail: self.run_crypten_party_jail,
        }

    def run_crypten_party_plan(self, msg: CryptenInitPlan) -> ObjectMessage:
        """Run crypten party according to the information received.

        Args:
            msg (CryptenInitPlan): should contain the rank_to_worker_id, world_size,
                                master_addr and master_port.

        Returns:
            An ObjectMessage containing the return value of the crypten function computed.
        """

        self.worker.rank_to_worker_id, world_size, master_addr, master_port = msg.crypten_context

        # TODO Change this, we need a way to handle multiple plan definitions
        plans = self.worker.search("crypten_plan")
        assert len(plans) == 1

        plan = plans[0].get()

        rank = None
        for r, worker_id in self.worker.rank_to_worker_id.items():
            if worker_id == self.worker.id:
                rank = r
                break

        assert rank is not None

        return_value = run_party(plan, rank, world_size, master_addr, master_port, (), {})
        return ObjectMessage(return_value)

    def run_crypten_party_jail(self, msg: CryptenInitJail):
        """Run crypten party according to the information received.

        Args:
            message (CryptenInitJail): should contain the rank, world_size,
                                    master_addr and master_port.

        Returns:
            An ObjectMessage containing the return value of the crypten function computed.
        """

        self.worker.rank_to_worker_id, world_size, master_addr, master_port = msg.crypten_context

        ser_func = msg.jail_runner
        onnx_model = msg.model
        crypten_model = None if onnx_model is None else utils.onnx_to_crypten(onnx_model)
        jail_runner = JailRunner.detail(ser_func, model=crypten_model)

        rank = None
        for r, worker_id in self.worker.rank_to_worker_id.items():
            if worker_id == self.worker.id:
                rank = r
                break

        assert rank is not None

        return_value = run_party(jail_runner, rank, world_size, master_addr, master_port, (), {})
        return ObjectMessage(return_value)
