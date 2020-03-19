import syft as sy
from syft.execution.state import State
from syft.serde import protobuf
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft.execution.placeholder import PlaceHolder

# assume _diff_state produced the same as here
# https://github.com/OpenMined/PySyft/blob/ryffel/syft-core/examples/experimental/FL%20Training%20Plan/Execute%20Plan.ipynb
# see step 7
# This serialization format will likely to change


def unserialize_model_params(bin: bin):
    """Unserializes model or checkpoint or diff stored in db to list of tensors"""
    state = StatePB()
    state.ParseFromString(bin)
    worker = sy.VirtualWorker(hook=None)
    state = protobuf.serde._unbufferize(worker, state)
    model_params = state.tensors()
    return model_params


def serialize_model_params(params):
    """Serializes list of tensors into State/protobuf"""
    model_params_state = State(
        owner=None,
        state_placeholders=[PlaceHolder().instantiate(param) for param in params],
    )

    # make fake local worker for serialization
    worker = sy.VirtualWorker(hook=None)

    pb = protobuf.serde._bufferize(worker, model_params_state)
    serialized_state = pb.SerializeToString()

    return serialized_state
