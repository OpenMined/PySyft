from typing import Dict
import types

from syft.workers.base import BaseWorker

from syft.messaging.message import ObjectMessage
from syft.messaging.message import CryptenInitPlan
from syft.messaging.message import CryptenInitJail

from syft.frameworks.crypten import run_party


def get_worker_from_rank(worker: BaseWorker, rank: int) -> BaseWorker:
    assert hasattr(worker, "rank_to_worker_id"), "First need to call run_crypten_party"
    return worker._get_worker_based_on_id(worker.rank_to_worker_id[rank])


def _set_rank_to_worker_id(worker: BaseWorker, rank_to_worker_id: Dict[int, int]) -> None:
    worker.rank_to_worker_id = rank_to_worker_id


def run_crypten_party_plan(self, message: CryptenInitPlan):
    """Run crypten party according to the information received.

        Args:
            message (CryptenInitPlan): should contain the rank, world_size, master_addr and master_port.

        Returns:
            An ObjectMessage containing the return value of the crypten function computed.
        """

    self.rank_to_worker_id, world_size, master_addr, master_port = message.crypten_context

    plans = self.search("crypten_plan")
    assert len(plans) == 1

    plan = plans[0].get()

    rank = None
    for r, worker_id in self.rank_to_worker_id.items():
        if worker_id == self.id:
            rank = r
            break

    assert rank is not None

    return_value = run_party(plan, rank, world_size, master_addr, master_port, (), {})
    return ObjectMessage(return_value)


def run_crypten_party_jail(self, message: CryptenInitJail):
    """Run crypten party according to the information received.

        Args:
            message (CryptenInitJail): should contain the rank, world_size, master_addr and master_port.

        Returns:
            An ObjectMessage containing the return value of the crypten function computed.
        """
    from syft.frameworks.crypten.jail import JailRunner
    from syft.frameworks.crypten import utils

    self.rank_to_worker_id, world_size, master_addr, master_port = message.crypten_context
    ser_func = message.jail_runner
    onnx_model = message.model
    crypten_model = None if onnx_model is None else utils.onnx_to_crypten(onnx_model)
    jail_runner = JailRunner.detail(ser_func, model=crypten_model)

    rank = None
    for r, worker_id in self.rank_to_worker_id.items():
        if worker_id == self.id:
            rank = r
            break

    assert rank is not None

    return_value = run_party(jail_runner, rank, world_size, master_addr, master_port, (), {})
    return ObjectMessage(return_value)


def add_support_to_workers(worker):
    worker._message_router[CryptenInitPlan] = types.MethodType(run_crypten_party_plan, worker)
    worker._message_router[CryptenInitJail] = types.MethodType(run_crypten_party_jail, worker)

    for method in methods_to_add:
        setattr(worker, method.__name__, types.MethodType(method, worker))


def remove_support_from_workers(worker):
    for method in methods_to_add:
        delattr(worker, method.__name__)


methods_to_add = [
    run_crypten_party_plan,
    run_crypten_party_jail,
    get_worker_from_rank,
    _set_rank_to_worker_id,
]
