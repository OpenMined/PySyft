# syft relative
from ..federated import JSONDict
from ..federated.model_centric_fl_base import ModelCentricFLBase
from ..lib.python import List
from ..lib.torch.module import Module as SyModule
from ..proto.core.plan.plan_pb2 import Plan
from .model_serialization import deserialize_model_params
from .model_serialization import wrap_model_params


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

        # store raw tensors only (not nn.Parameters, no grad)
        # TODO migrate to syft-core protobufs
        params = []
        model_parameters = model.parameters()
        if model_parameters is not None:
            params = [getattr(p, "data", None) for p in model_parameters]
        serialized_model = self.hex_serialize(wrap_model_params(params))

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
        # TODO migrate to syft-core protobufs
        return deserialize_model_params(serialized_model)
