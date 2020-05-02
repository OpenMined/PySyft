from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import copy
import inspect
import io
import torch
import warnings

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.role import Role
from syft.execution.state import State
from syft.execution.tracing import trace

# from syft.execution.translation.abstract import AbstractPlanTranslator
# from syft.execution.translation.default import PlanTranslatorDefault
# from syft.execution.translation.torchscript import PlanTranslatorTorchscript
from syft.generic.frameworks import framework_packages
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkLayerModule
from syft.generic.object import AbstractObject
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.protocol_pb2 import Protocol as ProtocolPB


class func2protocol(object):
    """Decorator which converts a function to a protocol.

    Converts a function containing sequential pytorch code into
    a protocol object which can be sent to any arbitrary worker.

    This class should be used only as a decorator.
    """

    def __init__(self, args_shape=None, state=None):
        self.args_shape = args_shape
        self.state_tensors = state or tuple()
        # include_state is used to distinguish if the initial protocol is a function or a class:
        # if it's a function, then the state should be provided in the args, so include_state
        # will be true. And to know if it was indeed a function, we just need to see if a
        # "manual" state was provided.
        self.include_state = state is not None

    def __call__(self, protocol_function):
        protocol = Protocol(
            name=protocol_function.__name__,
            include_state=self.include_state,
            forward_func=protocol_function,
            state_tensors=self.state_tensors,
            id=sy.ID_PROVIDER.pop(),
            owner=sy.local_worker,
        )

        # Build the protocol automatically
        if self.args_shape:
            args_ = PlaceHolder.create_placeholders(self.args_shape)
            try:
                protocol.build(*args_)
            except TypeError as e:
                raise ValueError(
                    "Automatic build using @func2protocol failed!\nCheck that:\n"
                    " - you have provided the correct number of shapes in args_shape\n"
                    " - you have no simple numbers like int or float as args. If you do "
                    "so, please consider using a tensor instead."
                )
        return protocol


class Protocol(AbstractObject):
    """
    A Protocol stores a sequence of torch actions, just like a function.

    A Protocol is intended to store a sequence of torch actions, just like a function,
    but it allows to send this sequence of actions to remote workers and to keep a
    reference to it. This way, to compute remotely this sequence of actions on some remote
    input referenced through pointers, instead of sending multiple messages you need now to send a
    single message with the references of the protocol and the pointers.

    All arguments are optional.

    Args:
        name: the name of the name
        state: store the protocol tensors like model parameters
        include_state: if true, implies that the protocol is a function, else a class. If true, the
            state is re-integrated in the args to be accessed within the function
        is_built: state if the protocol has already been built.
        placeholders: dict of placeholders used in the protocol
        actions: list of commands (called actions)
        forward_func: the function to be transformed into a protocol
        state_tensors: a tuple of state elements. It can be used to populate a state
        id: protocol id
        owner: protocol owner
        tags: protocol tags
        description: protocol description
    """

    # _build_translators = []

    def __init__(
        self,
        name: str = None,
        include_state: bool = False,
        is_built: bool = False,
        forward_func=None,
        state_tensors=[],
        roles: Dict[str, Role] = {},
        input_repartition: List[str] = [],
        output_repartition: List[str] = [],
        # General kwargs
        id: Union[str, int] = None,
        owner: "sy.workers.BaseWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        AbstractObject.__init__(self, id, owner, tags, description, child=None)

        # Protocol instance info
        self.name = name or self.__class__.__name__

        self.roles = roles
        self.input_repartition = input_repartition
        self.output_repartition = output_repartition

        # if role is None:
        #     for st in state_tensors:
        #         self.role.register_state_tensor(st, owner)

        self.include_state = include_state
        self.is_building = False
        self.state_attributes = {}
        self.is_built = is_built
        self.torchscript = None
        self.tracing = False

        # The protocol has not been sent so it has no reference to remote locations
        self.pointers = dict()

        if not hasattr(self, "forward"):
            self.forward = forward_func or None

        self.__name__ = self.__repr__()  # For PyTorch jit tracing compatibility

        # List of available translations
        # self.translations = []

    # @property
    # def state(self):
    #     return self.role.state

    # def parameters(self):
    #     """
    #     This is defined to match the torch api of nn.Module where .parameters() return the model tensors / parameters
    #     """
    #     if self.state is not None:
    #         return self.state.tensors()
    #     else:
    #         return []

    def get_role_for_owner(self, owner):
        if owner.id not in self.roles:
            self.roles[owner.id] = Role()
        return self.roles[owner.id]

    def build(self, *args):
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
        self.roles = {}
        self.input_repartition = []
        self.output_repartition = []

        # Enable tracing
        self.toggle_tracing(True)
        self.is_building = True

        # Run once to build the protocol
        ph_args = tuple()
        for arg in args:
            arg_role = self.get_role_for_owner(arg.owner)

            ph_arg = PlaceHolder.create_from(arg, owner=arg.owner, role=arg_role, tracing=True)
            # Register inputs in role
            arg_role.register_input(ph_arg)

            self.input_repartition.append(arg.owner.id)

            ph_args += (ph_arg,)

        # Add state to args if needed
        # if self.include_state:
        #     ph_args += (self.state,)

        # with trace(framework_packages["torch"], self.role, self.owner) as wrapped_torch:
        # Look for framework kwargs
        # framework_kwargs = {}
        # forward_args = inspect.getargspec(self.forward).args
        # if "torch" in forward_args:
        #     framework_kwargs["torch"] = wrapped_torch

        # results = self.forward(*args, **framework_kwargs)
        results = self.forward(*ph_args)

        # Disable tracing
        self.toggle_tracing(False)
        self.is_building = False

        # Register outputs in roles
        for result in results:
            if isinstance(result, PlaceHolder):
                result_role = self.get_role_for_owner(result.owner)
                result_role.register_output(result)

                self.output_repartition.append(result.owner.id)

        self.is_built = True

        # Build registered translations
        # for translator in Protocol._build_translators:
        #     try:
        #         self.add_translation(translator)
        #         self.translations.append(translator)
        #     except:
        #         warnings.warn(f"Failed to translate Protocol with {translator}")

        return results

    def toggle_tracing(self, value=None):
        self.tracing = value if value is not None else not self.tracing
        # self.state.tracing = self.tracing
        for role in self.roles.values():
            for ph in role.placeholders.values():
                ph.tracing = self.tracing

    def copy(self):
        """Creates a copy of a protocol."""
        protocol_copy = Protocol(
            name=self.name,
            roles={role_id: role.copy() for role_id, role in self.roles.items()},
            input_repartition=self.input_repartition,
            output_repartition=self.output_repartition,
            include_state=self.include_state,
            is_built=self.is_built,
            id=sy.ID_PROVIDER.pop(),
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        protocol_copy.torchscript = self.torchscript

        return protocol_copy

    # def __setattr__(self, name, value):
    #     """Add new tensors or parameter attributes to the state and register them
    #     in the owner's registry
    #     """
    #     if isinstance(value, torch.jit.ScriptModule):
    #         object.__setattr__(self, name, value)
    #     elif isinstance(value, FrameworkTensor):
    #         self.role.register_state_tensor(value, self.owner)
    #         self.state_attributes[name] = value
    #     elif isinstance(value, FrameworkLayerModule):
    #         for param in value.parameters():
    #             self.role.register_state_tensor(param, self.owner)
    #         self.state_attributes[name] = value
    #     else:
    #         object.__setattr__(self, name, value)

    # def __getattr__(self, name):
    #     if name not in self.state_attributes:
    #         raise AttributeError("State attribute not found.")

    #     value = self.state_attributes[name]
    #     if not self.is_building:
    #         return value

    #     if isinstance(value, FrameworkTensor):
    #         return self.role.placeholders[value.id]
    #     elif isinstance(value, FrameworkLayerModule):
    #         # We need to deepcopy here otherwise the real layer is modified when the Protocol is being built
    #         copied_layer = copy.deepcopy(value)
    #         for copied_param, param in zip(copied_layer.named_parameters(), value.parameters()):
    #             (copied_name, _) = copied_param
    #             copied_layer._parameters[copied_name] = self.role.placeholders[param.id]

    #         return copied_layer

    def __call__(self, *args):
        """
        Calls a protocol execution with some arguments.

        When possible, run the original function to improve efficiency. When
        it's not, for example if you fetched the protocol from a remote worker,
        then run it from the tape of actions:
        - Instantiate input placeholders
        - for each recorded action, run the action on the placeholders
          and use the result(s) to instantiate to appropriate placeholder.
        - Return the instantiation of all the output placeholders.
        """
        if self.forward is not None:
            if self.include_state:
                args = (*args, self.state)
            return self.forward(*args)
        else:
            results_per_role = {}
            for role_id, role in self.roles.items():
                args_for_role = [
                    arg for ind, arg in enumerate(args) if self.input_repartition[ind] == role_id
                ]
                results_per_role[role_id] = list(role.execute(args_for_role))

            results = ()
            for role_id in self.output_repartition:
                results += (results_per_role[role_id].pop(0),)
            return results

    def run(self, args_: Tuple, result_ids: List[Union[str, int]]):
        """Controls local or remote protocol execution.
        If the protocol doesn't have the protocol built, first build it using the original function.

        Args:
            args_: Arguments used to run protocol.
            result_ids: List of ids where the results will be stored.
        """
        # TODO: can we reuse result_ids?
        return self.__call__(*args_)

    # def send(self, *locations: AbstractWorker, force=False):
    #     """Send protocol to locations.

    #     If the protocol was not built locally it will raise an exception.
    #     If `force` = true protocol is going to be sent either way.

    #     Args:
    #         locations: List of workers.
    #         force: A boolean indicating if this action should be forced.
    #     """
    #     if not self.is_built and not force:
    #         raise RuntimeError("A protocol needs to be built before being sent to a worker.")

    #     if len(locations) == 1:
    #         location = locations[0]

    #         # Check if protocol was already sent at the location
    #         if location in self.pointers:
    #             return self.pointers[location]

    #         # Send the Protocol
    #         pointer = self.owner.send(self, workers=location)

    #         self.pointers[location] = pointer
    #     else:
    #         ids_at_location = []
    #         for location in locations:
    #             if location in self.pointers:
    #                 # Use the pointer that was already sent
    #                 pointer = self.pointers[location]
    #             else:
    #                 # Send the Protocol
    #                 pointer = self.owner.send(self, workers=location)

    #                 self.pointers[location] = pointer

    #             ids_at_location.append(pointer.id_at_location)

    #         pointer = sy.PointerProtocol(location=locations, id_at_location=ids_at_location)

    #     return pointer

    # def get_args_shape(self):
    #     """Returns input tensors shapes"""
    #     if not self.is_built:
    #         raise RuntimeError("A protocol needs to be built before input shapes can be known.")

    #     return [ph.expected_shape for ph in self.role.input_placeholders()]

    # @staticmethod
    # def register_build_translator(translator: "AbstractPlanTranslator"):
    #     Plan._build_translators.append(translator)

    # def add_translation(self, plan_translator: "AbstractPlanTranslator"):
    #     return plan_translator(self).translate()

    # def remove_translation(self, plan_translator: "AbstractPlanTranslator" = PlanTranslatorDefault):
    #     plan_translator(self).remove()
    #     return self

    # def get_(self):
    #     self.state.get_()
    #     return self

    # get = get_

    def get_pointers(self):
        return self.pointers

    def fix_precision_(self, *args, **kwargs):
        self.state.fix_precision_(*args, **kwargs)
        return self

    fix_precision = fix_prec_ = fix_prec = fix_precision_

    def float_precision_(self):
        self.state.float_precision_()
        return self

    float_precision = float_prec_ = float_prec = float_precision_

    def share_(self, *args, **kwargs):
        self.state.share_(*args, **kwargs)
        return self

    share = share_

    # def create_pointer(
    #     self, owner, garbage_collect_data, location=None, id_at_location=None, tags=None, **kwargs
    # ):
    #     """
    #     Create a pointer to the protocol

    #     Args:
    #         owner: the owner of the pointer
    #         garbage_collect_data: if true, when the pointer is deleted, the remote target is garbaged collected
    #         location: the location of the pointer
    #         id_at_location: the remote id at location
    #         tags: the tags inherited from the Protocol

    #     Returns:
    #         PointerProtocol: pointer to the protocol
    #     """
    #     return PointerProtocol(
    #         owner=owner,
    #         location=location or self.owner,
    #         id_at_location=id_at_location or self.id,
    #         garbage_collect_data=garbage_collect_data,
    #         tags=tags,
    #     )

    @staticmethod
    def replace_non_instanciated_placeholders(protocol: "Protocol") -> "Protocol":
        # Replace non-instanciated placeholders from protocol.placeholders by instanciated placeholders
        # from state.state_placeholders
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
            sy.serde.msgpack.serde._simplify(worker, protocol.input_repartition),
            sy.serde.msgpack.serde._simplify(worker, protocol.output_repartition),
            sy.serde.msgpack.serde._simplify(worker, protocol.include_state),
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
        (
            id_,
            name,
            roles,
            input_repartition,
            output_repartition,
            include_state,
            tags,
            description,
        ) = protocol_tuple

        id_ = sy.serde.msgpack.serde._detail(worker, id_)
        name = sy.serde.msgpack.serde._detail(worker, name)
        roles = sy.serde.msgpack.serde._detail(worker, roles)
        input_repartition = sy.serde.msgpack.serde._detail(worker, input_repartition)
        output_repartition = sy.serde.msgpack.serde._detail(worker, output_repartition)
        tags = sy.serde.msgpack.serde._detail(worker, tags)
        description = sy.serde.msgpack.serde._detail(worker, description)

        return sy.Protocol(
            id=id_,
            name=name,
            owner=worker,
            roles=roles,
            input_repartition=input_repartition,
            output_repartition=output_repartition,
            include_state=include_state,
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

        protobuf_protocol.input_repartition.extend(protocol.input_repartition)
        protobuf_protocol.output_repartition.extend(protocol.output_repartition)

        protobuf_protocol.include_state = protocol.include_state
        protobuf_protocol.tags.extend(protocol.tags)

        if protocol.description:
            protobuf_protocol.description = protocol.description

        return protobuf_protocol

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_protocol: ProtocolPB) -> "Protocol":
        """This function reconstructs a Protocol object given its attributes in the form of a Protobuf message
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

        input_repartition = protobuf_protocol.input_repartition
        output_repartition = protobuf_protocol.output_repartition

        include_state = protobuf_protocol.include_state
        tags = set(protobuf_protocol.tags) if protobuf_protocol.tags else None
        description = protobuf_protocol.description if protobuf_protocol.description else None

        return Protocol(
            id=id_,
            name=name,
            roles=roles,
            input_repartition=input_repartition,
            output_repartition=output_repartition,
            include_state=include_state,
            is_built=True,
            owner=worker,
            tags=tags,
            description=description,
        )


# Auto-register Protocol build-time translations
# Protocol.register_build_translator(PlanTranslatorTorchscript)
