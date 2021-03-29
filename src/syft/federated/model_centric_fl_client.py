# syft absolute
from syft.proto.core.plan.plan_pb2 import Plan

# syft relative
from ..federated import JSONDict
from ..federated.model_centric_fl_base import ModelCentricFLBase
from ..lib.python import List
from ..lib.torch.module import Module as SyModule
from ..proto.lib.python.list_pb2 import List as ListPB


class ModelCentricFLClient(ModelCentricFLBase):
    def host_federated_training(
        self,
        model: SyModule,
        client_plans: JSONDict,
        client_protocols: JSONDict,
        client_config: JSONDict,
        server_averaging_plan: Plan,
        server_config: JSONDict,
    ) -> JSONDict:

        serialized_model = self.hex_serialize(List(model.parameters()))
        serialized_plans = self._serialize_dict_values(client_plans)
        serialized_protocols = self._serialize_dict_values(client_protocols)
        serialized_avg_plan = self.hex_serialize(server_averaging_plan)

        # "model-centric/host-training" request body
        message = {
            "type": "model-centric/host-training",
            "data": {
                "model": serialized_model,
                "plans": serialized_plans,
                "protocols": serialized_protocols,
                "averaging_plan": serialized_avg_plan,
                "client_config": client_config,
                "server_config": server_config,
            },
        }

        return self._send_msg(message)

    def retrieve_model(
        self, name: str, version: str, checkpoint: str = "latest"
    ) -> List:
        params = {
            "name": name,
            "version": version,
            "checkpoint": checkpoint,
        }
        serialized_model = self._send_http_req(
            "GET", "/model-centric/retrieve-model", params
        )
        return self._unserialize(serialized_model, ListPB)
