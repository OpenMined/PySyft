from typing import List

import torch

import syft as sy
from syft.generic.abstract.syft_serializable import SyftSerializable
from syft.workers.abstract import AbstractWorker
from syft_proto.execution.v1.state_pb2 import State as StatePB
from syft_proto.execution.v1.state_tensor_pb2 import StateTensor as StateTensorPB
from syft_proto.types.torch.v1.parameter_pb2 import Parameter as ParameterPB


class State(SyftSerializable):
    """The State is a Plan attribute and is used to send tensors along functions.

    It references Plan tensor or parameters attributes using their name, and make
    sure they are provided to remote workers who are sent the Plan.
    """

    def __init__(self, state_placeholders=None):
        self.state_placeholders = state_placeholders or []
        self.tracing = False

    def __str__(self):
        """Returns the string representation of the State."""
        out = "<"
        out += "State:"
        for state_placeholder in self.state_placeholders:
            out += f" {state_placeholder}"
        out += ">"
        return out

    def __repr__(self):
        return self.__str__()

    def tensors(self) -> List:
        """
        Fetch and return all the state elements.
        """
        return [placeholder.child for placeholder in self.state_placeholders]

    def copy(self) -> "State":
        return State(self.state_placeholders.copy())

    def read(self):
        """
        Return state tensors that are from this plan specifically, but not those
        of plans including in this plan.
        If run while a plan is building, declare all the state tensors to the plan
        currently building.
        """
        if self.tracing:
            return list(self.state_placeholders)
        else:
            return [ph.child for ph in self.state_placeholders]

    @staticmethod
    def create_grad_if_missing(tensor):
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is None:
            o = tensor.sum()
            o.backward()
            if tensor.grad is not None:
                tensor.grad -= tensor.grad

    def fix_precision_(self, *args, **kwargs):
        for tensor in self.tensors():
            self.create_grad_if_missing(tensor)
            tensor.fix_precision_(*args, **kwargs)

    def float_precision_(self):
        for tensor in self.tensors():
            tensor.float_precision_()

    def share_(self, *args, **kwargs):
        for tensor in self.tensors():
            self.create_grad_if_missing(tensor)
            tensor.share_(*args, **kwargs)

    def get_(self):
        """
        Get functionality that can only be used when getting back state
        elements converted to additive shared tensors. Other than this,
        you shouldn't need to the get the state separately.
        """
        # TODO Make it only valid for AST
        for tensor in self.tensors():
            tensor.get_()

    @staticmethod
    def simplify(worker: AbstractWorker, state: "State") -> tuple:
        """
        Simplify the plan's state when sending a plan
        """
        return (
            sy.serde.msgpack.serde._simplify(worker, state.state_placeholders),
            sy.serde.msgpack.serde._simplify(worker, state.tensors()),
        )

    @staticmethod
    def detail(worker: AbstractWorker, state_tuple: tuple) -> "State":
        """
        Reconstruct the plan's state from the state elements and supposed
        ids.
        """
        state_placeholders, state_elements = state_tuple

        state_placeholders = sy.serde.msgpack.serde._detail(worker, state_placeholders)
        state_elements = sy.serde.msgpack.serde._detail(worker, state_elements)

        for state_element in state_elements:
            worker.register_obj(state_element, obj_id=state_element.id)

        for state_placeholder, state_element in zip(state_placeholders, state_elements):
            state_placeholder.instantiate(state_element)

        state = State(state_placeholders)
        return state

    @staticmethod
    def bufferize(worker: AbstractWorker, state: "State") -> StatePB:
        """
        Serialize the State to Protobuf message
        """
        protobuf_state = StatePB()

        protobuf_placeholders = [
            sy.serde.protobuf.serde._bufferize(worker, placeholder)
            for placeholder in state.state_placeholders
        ]
        protobuf_state.placeholders.extend(protobuf_placeholders)

        state_tensors = []
        for tensor in state.tensors():
            protobuf_tensor = sy.serde.protobuf.serde._bufferize(worker, tensor)
            state_tensor = StateTensorPB()
            if type(protobuf_tensor) == ParameterPB:
                state_tensor.torch_param.CopyFrom(
                    sy.serde.protobuf.serde._bufferize(worker, tensor)
                )
            else:
                state_tensor.torch_tensor.CopyFrom(
                    sy.serde.protobuf.serde._bufferize(worker, tensor)
                )
            state_tensors.append(state_tensor)

        protobuf_state.tensors.extend(state_tensors)

        return protobuf_state

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_state: StatePB) -> "State":
        """
        Reconstruct the plan's state from the state elements and supposed
        ids.
        """
        state_placeholders = protobuf_state.placeholders
        state_elements = protobuf_state.tensors

        state_placeholders = [
            sy.serde.protobuf.serde._unbufferize(worker, placeholder)
            for placeholder in protobuf_state.placeholders
        ]

        state_elements = []
        for protobuf_tensor in protobuf_state.tensors:
            tensor = getattr(protobuf_tensor, protobuf_tensor.WhichOneof("tensor"))
            state_elements.append(sy.serde.protobuf.serde._unbufferize(worker, tensor))

        for state_element in state_elements:
            worker.register_obj(state_element, obj_id=state_element.id)

        for state_placeholder, state_element in zip(state_placeholders, state_elements):
            state_placeholder.instantiate(state_element)

        state = State(state_placeholders)
        return state

    @staticmethod
    def get_protobuf_schema() -> StatePB:
        return StatePB
