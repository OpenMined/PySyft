from typing import List
from typing import Tuple
from typing import Union

import copy
import inspect
import io
import torch
import traceback
import warnings

import syft as sy
from syft.execution.placeholder import PlaceHolder
from syft.execution.role import Role
from syft.execution.state import State
from syft.execution.tracing import trace
from syft.generic.frameworks import framework_packages
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkLayerModule
from syft.generic.object import AbstractObject
from syft.generic.pointers.pointer_protocol import PointerProtocol
from syft.workers.abstract import AbstractWorker

from syft_proto.execution.v1.protocol_pb2 import Protocol as ProtocolPB


class func2protocol(object):
    """Decorator which converts a function to a protocol.

    Converts a function containing sequential pytorch code into
    a protocol object which can be sent to any arbitrary worker.

    This class should be used only as a decorator.
    """

    def __init__(self, args_shape=None, state=None):
        # print(36, 'proto __init__',args_shape)
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
        print(57, 'proto __call__',self.args_shape)
        if self.args_shape:
            args_ = PlaceHolder.create_placeholders(self.args_shape)
            try:
                print(60, 'proto try',*args_)
                protocol.build(*args_)
            except TypeError as e:
                tb = traceback.format_exc()
                print(65, 'proto', tb)
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

    def __init__(
        self,
        name: str = None,
        include_state: bool = False,
        is_built: bool = False,
        forward_func=None,
        state_tensors=[],
        role: Role = None,
        # General kwargs
        id: Union[str, int] = None,
        owner: "sy.workers.BaseWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        AbstractObject.__init__(self, id, owner, tags, description, child=None)

        # Protocol instance info
        self.name = name or self.__class__.__name__

        self.role = role or Role()

        if role is None:
            for st in state_tensors:
                self.role.register_state_tensor(st, owner)

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

    @property
    def state(self):
        return self.role.state

    @property
    def actions(self):
        return self.role.actions

    def parameters(self):
        """
        This is defined to match the torch api of nn.Module where .parameters() return the model tensors / parameters
        """
        if self.state is not None:
            return self.state.tensors()
        else:
            return []

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

        # Enable tracing
        self.toggle_tracing(True)
        self.is_building = True

        # Run once to build the protocol
        args = tuple(
            PlaceHolder.create_from(arg, owner=sy.local_worker, role=self.role, tracing=True)
            for arg in args
        )
        print(180, 'protocol.py', args)

        # Add state to args if needed
        if self.include_state:
            args += (self.state,)

        with trace(framework_packages["torch"], self.role, self.owner) as wrapped_torch:
            # Look for framework kwargs
            framework_kwargs = {}
            forward_args = inspect.getfullargspec(self.forward).args
            print(190, forward_args, inspect.getfullargspec(self.forward))
            if "torch" in forward_args or "tensor" in forward_args:
                print(192, 'trace?')
                framework_kwargs["torch"] = wrapped_torch
                # framework_kwargs["tensor"] = wrapped_torch
                print(194, wrapped_torch, *args)
                print(framework_kwargs)

            results = self.forward(*args)  #
            if isinstance(results, PlaceHolder) or (
                isinstance(results, (list, tuple))
                and any(isinstance(r, PlaceHolder) for r in results)
            ):
                cmd = ('send', None, args, framework_kwargs)
                # if use framework_kwargs => send() got unexpected keyword
                # if use None => argument after ** must be a mapping
                # if use {} => syft/generic/pointers/multi_pointer.py:120: IndexError => index out of range
                results.handle_func_command(cmd)

        # Disable tracing
        self.toggle_tracing(False)
        self.is_building = False

        # Register inputs in role
        self.role.register_inputs(args)

        # Register outputs in role
        self.role.register_outputs(results)

        self.is_built = True

        return results

    def toggle_tracing(self, value=None):
        self.tracing = value if value is not None else not self.tracing
        self.state.tracing = self.tracing
        for ph in self.role.placeholders.values():
            ph.tracing = self.tracing

    def copy(self):
        """Creates a copy of a protocol."""
        protocol_copy = Protocol(
            name=self.name,
            role=self.role.copy(),
            include_state=self.include_state,
            is_built=self.is_built,
            id=sy.ID_PROVIDER.pop(),
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        protocol_copy.torchscript = self.torchscript

        return protocol_copy

    def __setattr__(self, name, value):
        """Add new tensors or parameter attributes to the state and register them
        in the owner's registry
        """
        if isinstance(value, torch.jit.ScriptModule):
            object.__setattr__(self, name, value)
        elif isinstance(value, FrameworkTensor):
            self.role.register_state_tensor(value, self.owner)
            self.state_attributes[name] = value
        elif isinstance(value, FrameworkLayerModule):
            for param in value.parameters():
                self.role.register_state_tensor(param, self.owner)
            self.state_attributes[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name not in self.state_attributes:
            raise AttributeError("State attribute not found.")

        value = self.state_attributes[name]
        if not self.is_building:
            return value

        if isinstance(value, FrameworkTensor):
            return self.role.placeholders[value.id]
        elif isinstance(value, FrameworkLayerModule):
            # We need to deepcopy here otherwise the real layer is modified when the Protocol is being built
            copied_layer = copy.deepcopy(value)
            for copied_param, param in zip(copied_layer.named_parameters(), value.parameters()):
                (copied_name, _) = copied_param
                copied_layer._parameters[copied_name] = self.role.placeholders[param.id]

            return copied_layer

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
            return self.role.execute(args)

    def run(self, args_: Tuple, result_ids: List[Union[str, int]]):
        """Controls local or remote protocol execution.
        If the protocol doesn't have the protocol built, first build it using the original function.

        Args:
            args_: Arguments used to run protocol.
            result_ids: List of ids where the results will be stored.
        """
        # TODO: can we reuse result_ids?
        return self.__call__(*args_)

    def send(self, *locations: AbstractWorker, force=False) -> PointerProtocol:
        """Send protocol to locations.

        If the protocol was not built locally it will raise an exception.
        If `force` = true protocol is going to be sent either way.

        Args:
            locations: List of workers.
            force: A boolean indicating if this action should be forced.
        """
        if not self.is_built and not force:
            raise RuntimeError("A protocol needs to be built before being sent to a worker.")

        if len(locations) == 1:
            location = locations[0]

            # Check if protocol was already sent at the location
            if location in self.pointers:
                return self.pointers[location]

            # Send the Protocol
            pointer = self.owner.send(self, workers=location)

            self.pointers[location] = pointer
        else:
            ids_at_location = []
            for location in locations:
                if location in self.pointers:
                    # Use the pointer that was already sent
                    pointer = self.pointers[location]
                else:
                    # Send the Protocol
                    pointer = self.owner.send(self, workers=location)

                    self.pointers[location] = pointer

                ids_at_location.append(pointer.id_at_location)

            pointer = sy.PointerProtocol(location=locations, id_at_location=ids_at_location)

        return pointer

    def get_args_shape(self):
        """Returns input tensors shapes"""
        if not self.is_built:
            raise RuntimeError("A protocol needs to be built before input shapes can be known.")

        return [ph.expected_shape for ph in self.role.input_placeholders()]

    def get_(self):
        self.state.get_()
        return self

    get = get_

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

    def create_pointer(
        self, owner, garbage_collect_data, location=None, id_at_location=None, tags=None, **kwargs
    ):
        """
        Create a pointer to the protocol

        Args:
            owner: the owner of the pointer
            garbage_collect_data: if true, when the pointer is deleted, the remote target is garbaged collected
            location: the location of the pointer
            id_at_location: the remote id at location
            tags: the tags inherited from the Protocol

        Returns:
            PointerProtocol: pointer to the protocol
        """
        return PointerProtocol(
            owner=owner,
            location=location or self.owner,
            id_at_location=id_at_location or self.id,
            garbage_collect_data=garbage_collect_data,
            tags=tags,
        )

    def __str__(self):
        """Returns the string representation of Protocol."""
        out = "<"
        out += str(type(self)).split("'")[1].split(".")[-1]
        out += " " + str(self.name)
        out += " id:" + str(self.id)
        out += " owner:" + str(self.owner.id)

        if self.tags is not None and len(self.tags):
            out += " Tags:"
            for tag in self.tags:
                out += " " + str(tag)

        if self.is_built:
            out += " built"

        out += ">"
        out += "\n"
        _self = self

        # out += f"def {self.name}("
        # out += ", ".join(f"arg_{extract_tag(p)}" for p in self.find_placeholders("input"))
        # out += "):\n"
        # for action in self.actions:
        #     line = "    "
        #     if action.return_ids is not None:
        #         if isinstance(action.return_ids, PlaceHolder):
        #             tag = extract_tag(action.return_ids)
        #             line += f"_{tag} = "
        #         elif isinstance(action.return_ids, tuple):
        #             line += (
        #                 ", ".join(
        #                     f"_{extract_tag(o)}" if isinstance(o, PlaceHolder) else str(o)
        #                     for o in action.return_ids
        #                 )
        #                 + " = "
        #             )
        #         else:
        #             line += str(action.return_ids) + " = "
        #     if action.target is not None:
        #         line += f"_{extract_tag(self.placeholders[action.target.value])}."
        #     line += action.name + "("
        #     line += ", ".join(
        #         f"_{extract_tag(arg)}" if isinstance(arg, PlaceHolder) else str(arg)
        #         for arg in action.args
        #     )
        #     if action.kwargs:
        #         line += ", " + ", ".join(f"{k}={w}" for k, w in action.kwargs.items())
        #     line += ")\n"
        #     out += line

        # out += "    return "
        # out += ", ".join(f"_{extract_tag(p)}" for p in self.find_placeholders("output"))

        return out

    def __repr__(self):
        return self.__str__()

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
        return (
            sy.serde.msgpack.serde._simplify(worker, protocol.id),
            sy.serde.msgpack.serde._simplify(worker, protocol.role),
            sy.serde.msgpack.serde._simplify(worker, protocol.include_state),
            sy.serde.msgpack.serde._simplify(worker, protocol.is_built),
            sy.serde.msgpack.serde._simplify(worker, protocol.name),
            sy.serde.msgpack.serde._simplify(worker, protocol.tags),
            sy.serde.msgpack.serde._simplify(worker, protocol.description),
            sy.serde.msgpack.serde._simplify(worker, protocol.torchscript),
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
        (id_, role, include_state, is_built, name, tags, description, torchscript) = protocol_tuple

        id_ = sy.serde.msgpack.serde._detail(worker, id_)
        role = sy.serde.msgpack.serde._detail(worker, role)
        name = sy.serde.msgpack.serde._detail(worker, name)
        tags = sy.serde.msgpack.serde._detail(worker, tags)
        description = sy.serde.msgpack.serde._detail(worker, description)
        torchscript = sy.serde.msgpack.serde._detail(worker, torchscript)

        protocol = sy.Protocol(
            role=role,
            include_state=include_state,
            is_built=is_built,
            id=id_,
            owner=worker,
            name=name,
            tags=tags,
            description=description,
        )

        protocol.torchscript = torchscript

        return protocol

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
        protobuf_protocol = ProtocolPB()

        sy.serde.protobuf.proto.set_protobuf_id(protobuf_protocol.id, protocol.id)

        protobuf_protocol.role.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, protocol.role))

        protobuf_protocol.include_state = protocol.include_state
        protobuf_protocol.is_built = protocol.is_built
        protobuf_protocol.name = protocol.name
        protobuf_protocol.tags.extend(protocol.tags)

        if protobuf_protocol.description:
            protobuf_protocol.description = protocol.description

        if protocol.torchscript:
            protobuf_protocol.torchscript = protocol.torchscript.save_to_buffer()

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

        role = sy.serde.protobuf.serde._unbufferize(worker, protobuf_protocol.role)

        name = protobuf_protocol.name
        tags = set(protobuf_protocol.tags) if protobuf_protocol.tags else None
        description = protobuf_protocol.description if protobuf_protocol.description else None

        protocol = Protocol(
            role=role,
            include_state=protobuf_protocol.include_state,
            is_built=protobuf_protocol.is_built,
            id=id_,
            owner=worker,
            name=name,
            tags=tags,
            description=description,
        )

        if protobuf_protocol.torchscript:
            torchscript = io.BytesIO(protobuf_protocol.torchscript)
            protocol.torchscript = torch.jit.load(torchscript)

        return protocol
