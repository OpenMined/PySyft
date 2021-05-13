# stdlib
from typing import List as TypeList
from typing import Union

# syft relative
from ...lib.python.list import List
from .placeholder import PlaceHolder
from .state import State
from .state import StatePB


def wrap_model_params(parameters: Union[List, TypeList]) -> State:
    """
    Wraps list of tensors in State object.
    """
    state = State(
        state_placeholders=[
            PlaceHolder(id=n).instantiate(p) for n, p in enumerate(parameters)
        ]
    )
    return state


def deserialize_model_params(proto: bytes) -> List:
    """
    Deserializes binary string (State protobuf) to List of tensors.
    """
    state_pb = StatePB()
    state_pb.ParseFromString(proto)
    state: State = State._proto2object(state_pb)
    return List(state.tensors())
