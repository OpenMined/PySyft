import syft

from syft.messaging.message import CryptenInitPlan
from syft.messaging.message import CryptenInitJail
from syft.messaging.message import ObjectMessage

from syft.frameworks import crypten as syft_crypten
from syft.frameworks.crypten.context import run_party
from syft.frameworks.crypten.jail import JailRunner
from syft.frameworks.crypten import utils

from syft.generic.abstract.message_handler import AbstractMessageHandler


class CryptenMessageHandler(AbstractMessageHandler):
    def __init__(self, object_store, worker):
        super().__init__(object_store)
        self.worker = worker

    def init_routing_table(self):
        return {
            CryptenInitPlan: self.run_crypten_party_plan,
            CryptenInitJail: self.run_crypten_party_jail,
        }

    def run_crypten_party_plan(self, msg: CryptenInitPlan) -> ObjectMessage:  # pragma: no cover
        """Run crypten party according to the information received.

        Args:
            msg (CryptenInitPlan): should contain the rank_to_worker_id, world_size,
                                master_addr and master_port.

        Returns:
            An ObjectMessage containing the return value of the crypten function computed.
        """

        rank_to_worker_id, world_size, master_addr, master_port = msg.crypten_context

        cid = syft.ID_PROVIDER.pop()
        syft_crypten.RANK_TO_WORKER_ID[cid] = rank_to_worker_id

        onnx_model = msg.model
        crypten_model = None if onnx_model is None else utils.onnx_to_crypten(onnx_model)

        # TODO Change this, we need a way to handle multiple plan definitions
        plans = self.worker.search("crypten_plan")
        assert len(plans) == 1

        plan = plans[0].get()

        rank = self._current_rank(rank_to_worker_id)
        assert rank is not None

        if crypten_model:
            args = (crypten_model,)
        else:
            args = ()

        return_value = run_party(cid, plan, rank, world_size, master_addr, master_port, args, {})
        # remove rank to id transaltion dict
        del syft_crypten.RANK_TO_WORKER_ID[cid]

        # Delete the plan at the end of the computation
        self.worker.de_register_obj(plan)

        return ObjectMessage(return_value)

    def run_crypten_party_jail(self, msg: CryptenInitJail):  # pragma: no cover
        """Run crypten party according to the information received.

        Args:
            message (CryptenInitJail): should contain the rank, world_size,
                                    master_addr and master_port.

        Returns:
            An ObjectMessage containing the return value of the crypten function computed.
        """

        rank_to_worker_id, world_size, master_addr, master_port = msg.crypten_context

        cid = syft.ID_PROVIDER.pop()
        syft_crypten.RANK_TO_WORKER_ID[cid] = rank_to_worker_id

        ser_func = msg.jail_runner
        onnx_model = msg.model
        crypten_model = None if onnx_model is None else utils.onnx_to_crypten(onnx_model)
        jail_runner = JailRunner.detail(ser_func, model=crypten_model)

        rank = self._current_rank(rank_to_worker_id)
        assert rank is not None

        return_value = run_party(
            cid, jail_runner, rank, world_size, master_addr, master_port, (), {}
        )
        # remove rank to id transaltion dict
        del syft_crypten.RANK_TO_WORKER_ID[cid]

        return ObjectMessage(return_value)

    def _current_rank(self, rank_to_worker_id):
        """Returns current rank based on worker_id."""
        rank = None
        for r, worker_id in rank_to_worker_id.items():
            if worker_id == self.worker.id:
                rank = r
                break
        return rank
