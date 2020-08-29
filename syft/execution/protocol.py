from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.role import Role
from syft.execution.role_assignments import RoleAssignments

from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.abstract import AbstractWorker
from syft.workers.virtual import VirtualWorker

from syft_proto.execution.v1.protocol_pb2 import Protocol as ProtocolPB


class func2protocol(object):
    """Decorator which converts a function to a protocol.

    Converts a function containing sequential pytorch code into
    a protocol object which can be sent to any arbitrary worker.

    This class should be used only as a decorator.
    """

    def __init__(self, roles: list = [], args_shape: dict = {}, states={}):
        self.role_names = roles
        self.args_shape = args_shape
        self.states = states

    def __call__(self, protocol_function):
        # create the roles present in decorator
        roles = {
            role_id: Role(worker=VirtualWorker(id=role_id, hook=sy.local_worker.hook))
            for role_id in self.role_names
        }
        for role_id, state_tensors in self.states.items():
            for tensor in state_tensors:
                roles[role_id].register_state_tensor(tensor)

        protocol = Protocol(
            name=protocol_function.__name__,
            forward_func=protocol_function,
            roles=roles,
            id=sy.ID_PROVIDER.pop(),
            owner=sy.local_worker,
        )

        try:
            protocol.build()
        except TypeError as e:
            raise ValueError(
                "Automatic build using @func2protocol failed!\nCheck that:\n"
                " - you have provided the correct number of shapes in args_shape\n"
                " - you have no simple numbers like int or float as args. If you do "
                "so, please consider using a tensor instead."
            )

        return protocol


class Protocol(AbstractSendable):
    """
    A Protocol stores a sequence of actions, just like a function.

    A Protocol is intended to store a sequence of actions, just like a function,
    but it allows to send this sequence of actions to remote workers and to keep a
    reference to it. This way, to compute remotely this sequence of actions on some remote
    input referenced through pointers, instead of sending multiple messages you need now to send a
    single message with the references of the protocol and the pointers.

    Specifically, a Protocol can contain a mix of ComputationAction and CommunicationAction and
    acts as a cross-worker. Use Plan for pure mathematical operations.

    All arguments are optional.

    Args:
        name: the name of the name
        is_built: state if the protocol has already been built.
        forward_func: the function to be transformed into a protocol
        id: protocol id
        owner: protocol owner
        tags: protocol tags
        description: protocol description
    """

    def __init__(
        self,
        name: str = None,
        is_built: bool = False,
        forward_func=None,
        roles: Dict[str, Role] = {},
        # General kwargs
        id: Union[str, int] = None,
        owner: "sy.workers.BaseWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        super().__init__(id, owner, tags, description, child=None)

        # Protocol instance info
        self.name = name or self.__class__.__name__

        self.roles = roles
        self.role_assignments = RoleAssignments(roles.keys())

        self.is_building = False
        self.is_built = is_built
        self.torchscript = None
        self.tracing = False

        if not hasattr(self, "forward"):
            self.forward = forward_func or None

        self.__name__ = self.__repr__()  # For PyTorch jit tracing compatibility

    def build(self):
        """Builds the protocol.

        First, run the function to be converted in a protocol in a context which
        activates the tracing and record the actions in trace.logs

        Second, store the result ids temporarily to helper ordering the output
        placeholders at return time

        Third, loop through the trace logs and replace the tensors found in the
        actions logged by PlaceHolders. Record those actions in
        protocol.actions

        Args:
            args: Input arguments to run the protocol
        """
        # Reset previous build
        for role in self.roles.values():
            role.reset()

        # Enable tracing
        self.toggle_tracing(True)
        self.is_building = True

        results = self.forward(*self.roles.values())

        # Disable tracing
        self.toggle_tracing(False)
        self.is_building = False

        if not isinstance(results, (tuple, list)):
            results = (results,)

        # Register outputs in roles
        for result in results:
            if isinstance(result, PlaceHolder):
                result.role.register_output(result)

        self.is_built = True

        return results

    def toggle_tracing(self, value=None):
        self.tracing = value if value is not None else not self.tracing
        # self.state.tracing = self.tracing
        for role in self.roles.values():
            role.tracing = value or not self.tracing
            for ph in role.placeholders.values():
                ph.tracing = self.tracing

    def copy(self):
        """Creates a copy of a protocol."""
        protocol_copy = Protocol(
            name=self.name,
            roles={role_id: role.copy() for role_id, role in self.roles.items()},
            is_built=self.is_built,
            id=sy.ID_PROVIDER.pop(),
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        protocol_copy.torchscript = self.torchscript

        return protocol_copy

    def __call__(self):
        """
        Run actions on the workers provided for each Role from the Role's tape of actions.
        """
        results_per_role = {}
        for role_id, role in self.roles.items():
            results_per_role[role_id] = role.execute()

        return results_per_role

    def run(self, args_: Tuple, result_ids: List[Union[str, int]]):
        """Controls local or remote protocol execution.
        If the protocol doesn't have the protocol built, first build it using the original function.

        Args:
            args_: Arguments used to run protocol.
            result_ids: List of ids where the results will be stored.
        """
        # TODO: can we reuse result_ids?
        return self.__call__(*args_)

    def assign(self, role_id, worker):
        """Assign a worker to the specified role."""
        self.role_assignments.assign(role_id, worker)

    def assign_roles(self, worker_dict):
        """Assign worker values to correspondent key role."""
        for role_id, worker in worker_dict.items():
            self.role_assignments.assign(role_id, worker)

    @staticmethod
    def replace_non_instanciated_placeholders(protocol: "Protocol") -> "Protocol":
        # Replace non-instanciated placeholders from protocol.placeholders by
        # instanciated placeholders from state.state_placeholders
        # NOTE Maybe state shouldn't contain instanciated placeholders but values directly?
        state_placeholders = {ph.id.value: ph for ph in protocol.state.state_placeholders}
        protocol.placeholders = {**protocol.placeholders, **state_placeholders}

        return protocol

    @staticmethod
    def simplify(worker: AbstractWorker, protocol: "Protocol") -> tuple:
        """
        This function takes the attributes of a Protocol and saves them in a tuple
        Args:
            worker (AbstractWorker): the worker doing the serialization
            protocol (Protocol): a Protocol object
        Returns:
            tuple: a tuple holding the unique attributes of the Protocol object

        """
        if not protocol.is_built:
            raise RuntimeError("A Protocol needs to be built before being serialized.")

        return (
            sy.serde.msgpack.serde._simplify(worker, protocol.id),
            sy.serde.msgpack.serde._simplify(worker, protocol.name),
            sy.serde.msgpack.serde._simplify(worker, protocol.roles),
            sy.serde.msgpack.serde._simplify(worker, protocol.tags),
            sy.serde.msgpack.serde._simplify(worker, protocol.description),
        )

    @staticmethod
    def detail(worker: AbstractWorker, protocol_tuple: tuple) -> "Protocol":
        """This function reconstructs a Protocol object given its attributes in the form of a tuple.
        Args:
            worker: the worker doing the deserialization
            protocol_tuple: a tuple holding the attributes of the Protocol
        Returns:
            protocol: a Protocol object
        """
        (id_, name, roles, tags, description) = protocol_tuple

        id_ = sy.serde.msgpack.serde._detail(worker, id_)
        name = sy.serde.msgpack.serde._detail(worker, name)
        roles = sy.serde.msgpack.serde._detail(worker, roles)
        tags = sy.serde.msgpack.serde._detail(worker, tags)
        description = sy.serde.msgpack.serde._detail(worker, description)

        return sy.Protocol(
            id=id_,
            name=name,
            owner=worker,
            roles=roles,
            is_built=True,
            tags=tags,
            description=description,
        )

    @staticmethod
    def bufferize(worker: AbstractWorker, protocol: "Protocol") -> ProtocolPB:
        """
        This function takes the attributes of a Protocol and saves them in a Protobuf message
        Args:
            worker (AbstractWorker): the worker doing the serialization
            protocol (Protocol): a Protocol object
        Returns:
            ProtocolPB: a Protobuf message holding the unique attributes of the Protocol object
        """
        if not protocol.is_built:
            raise RuntimeError("A Protocol needs to be built before being serialized.")

        protobuf_protocol = ProtocolPB()

        sy.serde.protobuf.proto.set_protobuf_id(protobuf_protocol.id, protocol.id)
        protobuf_protocol.name = protocol.name

        for role_id, role in protocol.roles.items():
            protobuf_protocol.roles.get_or_create(role_id).CopyFrom(
                sy.serde.protobuf.serde._bufferize(worker, role)
            )

        protobuf_protocol.tags.extend(protocol.tags)

        if protocol.description:
            protobuf_protocol.description = protocol.description

        return protobuf_protocol

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_protocol: ProtocolPB) -> "Protocol":
        """This function reconstructs a Protocol object given its attributes in the form
        of a Protobuf message

        Args:
            worker: the worker doing the deserialization
            protobuf_protocol: a Protobuf message holding the attributes of the Protocol
        Returns:
            protocol: a Protocol object
        """
        id_ = sy.serde.protobuf.proto.get_protobuf_id(protobuf_protocol.id)
        name = protobuf_protocol.name

        roles = {
            role_id: sy.serde.protobuf.serde._unbufferize(worker, role)
            for role_id, role in protobuf_protocol.roles.items()
        }

        tags = set(protobuf_protocol.tags) if protobuf_protocol.tags else None
        description = protobuf_protocol.description if protobuf_protocol.description else None

        return Protocol(
            id=id_,
            name=name,
            roles=roles,
            is_built=True,
            owner=worker,
            tags=tags,
            description=description,
        )

    @staticmethod
    def get_protobuf_schema() -> ProtocolPB:
        return ProtocolPB
