import re
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import torch

import syft as sy
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkLayerModule
from syft.generic.object import AbstractObject
from syft.generic.object_storage import ObjectStorage
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.messaging.message import Operation
from syft.messaging.plan.state import State
from syft.workers.abstract import AbstractWorker
from syft.frameworks.torch.tensors.interpreters.placeholder import PlaceHolder
from syft_proto.messaging.v1.message_pb2 import OperationMessage as OperationMessagePB
from syft_proto.messaging.v1.plan_pb2 import Plan as PlanPB


class func2plan(object):
    """Decorator which converts a function to a plan.

    Converts a function containing sequential pytorch code into
    a plan object which can be sent to any arbitrary worker.

    This class should be used only as a decorator.
    """

    def __init__(self, args_shape=None, state=None):
        self.args_shape = args_shape
        self.state_tensors = state or tuple()
        # include_state is used to distinguish if the initial plan is a function or a class:
        # if it's a function, then the state should be provided in the args, so include_state
        # will be true. And to know if it was indeed a function, we just need to see if a
        # "manual" state was provided.
        self.include_state = state is not None

    def __call__(self, plan_function):
        plan = Plan(
            name=plan_function.__name__,
            include_state=self.include_state,
            forward_func=plan_function,
            state_tensors=self.state_tensors,
            id=sy.ID_PROVIDER.pop(),
            owner=sy.local_worker,
        )

        # Build the plan automatically
        if self.args_shape:
            args = Plan._create_placeholders(self.args_shape)
            plan.build(*args)
        return plan


def method2plan(*args, **kwargs):
    raise SyntaxError(
        "method2plan is not supported anymore. Consider instead subclassing your object from sy.Plan."
    )


class Plan(AbstractObject, ObjectStorage):
    """
    A Plan stores a sequence of torch operations, just like a function.

    A Plan is intended to store a sequence of torch operations, just like a function,
    but it allows to send this sequence of operations to remote workers and to keep a
    reference to it. This way, to compute remotely this sequence of operations on some remote
    input referenced through pointers, instead of sending multiple messages you need now to send a
    single message with the references of the plan and the pointers.

    All arguments are optional.

    Args:
        name: the name of the name
        state: store the plan tensors like model parameters
        include_state: if true, implies that the plan is a function, else a class. If true, the
            state is re-integrated in the args to be accessed within the function
        is_built: state if the plan has already been built.
        placeholders: dict of placeholders used in the plan
        operations: list of commands (called operations)
        forward_func: the function to be transformed into a plan
        state_tensors: a tuple of state elements. It can be used to populate a state
        id: plan id
        owner: plan owner
        tags: plan tags
        description: plan description
    """

    def __init__(
        self,
        name: str = None,
        state: State = None,
        include_state: bool = False,
        is_built: bool = False,
        operations: List[Operation] = None,
        placeholders: Dict[Union[str, int], PlaceHolder] = None,
        forward_func=None,
        state_tensors=None,
        # General kwargs
        id: Union[str, int] = None,
        owner: "sy.workers.BaseWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        owner = owner or sy.local_worker
        AbstractObject.__init__(self, id, owner, tags, description, child=None)
        ObjectStorage.__init__(self)

        # Plan instance info
        self.name = name or self.__class__.__name__
        self.owner = owner

        self.operations = operations or []

        # Keep a local reference to all placeholders, stored by id
        self.placeholders = placeholders or {}
        # Incremental value to tag all placeholders with different tags
        self.var_count = 0

        self.state = state or State(owner=owner)
        # state_tensors are provided when plans are created using func2plan
        if state_tensors is not None:
            # we want to make sure in that case that the state is empty
            assert state is None
            for tensor in state_tensors:
                placeholder = sy.PlaceHolder(
                    tags={"#state", f"#{self.var_count + 1}"}, id=tensor.id
                )
                self.var_count += 1
                placeholder.instantiate(tensor)
                self.state.state_placeholders.append(placeholder)
                self.placeholders[tensor.id] = placeholder

        self.include_state = include_state
        self.is_built = is_built

        # The plan has not been sent so it has no reference to remote locations
        self.pointers = dict()

        if forward_func is not None:
            self.forward = forward_func
        elif self.is_built:
            self.forward = None

        self.__name__ = self.__repr__()  # For PyTorch jit tracing compatibility

    @staticmethod
    def _create_placeholders(args_shape):
        # In order to support -1 value in shape to indicate any dimension
        # we map -1 to 1 for shape dimensions.
        # TODO: A more complex strategy could be used
        mapped_shapes = []
        for shape in args_shape:
            if list(filter(lambda x: x < -1, shape)):
                raise ValueError("Invalid shape {}".format(shape))
            mapped_shapes.append(tuple(map(lambda y: 1 if y == -1 else y, shape)))

        return [sy.framework.hook.create_zeros(shape) for shape in mapped_shapes]

    @property
    def _known_workers(self):
        return self.owner._known_workers

    @property
    def location(self):
        raise AttributeError("Plan has no attribute location")

    # For backward compatibility
    @property
    def readable_plan(self):
        return self.operations

    def parameters(self):
        """
        This is defined to match the torch api of nn.Module where .parameters() return the model tensors / parameters
        """
        return self.state.tensors()

    def send_msg(self, *args, **kwargs):
        return self.owner.send_msg(*args, **kwargs)

    def request_obj(self, *args, **kwargs):
        return self.owner.request_obj(*args, **kwargs)

    def respond_to_obj_req(self, obj_id: Union[str, int]):
        """Returns the deregistered object from registry.

        Args:
            obj_id: A string or integer id of an object to look up.
        """

        obj = self.get_obj(obj_id)
        self.de_register_obj(obj)
        return obj

    def add_placeholder(self, tensor, node_type=None):
        """
        Create and register a new placeholder if not already existing (else return
        the existing one).

        The placeholder is tagged by a unique and incremental index for a given plan.

        Args:
            tensor: the tensor to replace with a placeholder
            node_type: Should be "input" or "output", used to tag like this: #<type>-*
        """
        if tensor.id not in self.placeholders.keys():
            placeholder = sy.PlaceHolder(tags={f"#{self.var_count + 1}"}, id=tensor.id)
            self.placeholders[tensor.id] = placeholder

            if node_type == "input":
                placeholder.tags.add(f"#input-{self._tmp_args_ids.index(tensor.id)}")
            elif node_type == "output":
                if tensor.id in self._tmp_result_ids:
                    placeholder.tags.add(f"#output-{self._tmp_result_ids.index(tensor.id)}")
            else:
                raise ValueError("node_type should be 'input' or 'output'.")

            self.var_count += 1

        return self.placeholders[tensor.id]

    def replace_with_placeholders(self, obj, **kw):
        """
        Replace in an object all FrameworkTensors with Placeholders
        """
        if isinstance(obj, (tuple, list)):
            r = [self.replace_with_placeholders(o, **kw) for o in obj]
            return type(obj)(r)
        elif isinstance(obj, dict):
            return {key: self.replace_with_placeholders(value, **kw) for key, value in obj.items()}
        elif isinstance(obj, FrameworkTensor):
            return self.add_placeholder(obj, **kw)
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif obj is None:
            return None
        else:
            raise TypeError(f"Type {type(obj)} not supported in plans args/kwargs")

    def find_placeholders(self, *search_tags):
        """
        Search method to retrieve placeholders used in the Plan using tag search.
        Retrieve all placeholders which have a tag containing at least one search_tag.

        Args:
            *search_tags: tuple of tags

        Returns:
            A list of placeholders found
        """
        results = []
        for placeholder in self.placeholders.values():
            for search_tag in search_tags:
                for tag in placeholder.tags:
                    match = re.search(f".*{search_tag}.*", tag)
                    if match is not None:
                        results.append(placeholder)

        return results

    def build(self, *args):
        """Builds the plan.

        First, run the function to be converted in a plan in a context which
        activates the tracing and record the operations in trace.logs

        Second, store the result ids temporarily to helper ordering the output
        placeholders at return time

        Third, loop through the trace logs and replace the tensors found in the
        operations logged by PlaceHolders. Record those operations in
        plan.operations

        Args:
            args: Input arguments to run the plan
        """

        self.owner.init_plan = self

        self._tmp_args_ids = [t.id for t in args if isinstance(t, FrameworkTensor)]

        with sy.hook.trace.enabled():
            # We usually have include_state==True for functions converted to plan
            # using @func2plan and we need therefore to add the state manually
            if self.include_state:
                results = self.forward(*args, self.state)
            else:
                results = self.forward(*args)

        results = (results,) if not isinstance(results, tuple) else results
        self._tmp_result_ids = [t.id for t in results if isinstance(t, FrameworkTensor)]

        for log in sy.hook.trace.logs:
            command, response = log
            command_placeholders, return_placeholders = (
                self.replace_with_placeholders(command, node_type="input"),
                self.replace_with_placeholders(response, node_type="output"),
            )
            # We're cheating a bit here because we put placeholders instead of return_ids
            operation = Operation(*command_placeholders, return_ids=return_placeholders)
            self.operations.append(operation)

        sy.hook.trace.clear()
        del self._tmp_result_ids
        del self._tmp_args_ids

        self.is_built = True
        self.owner.init_plan = None

    def copy(self):
        """Creates a copy of a plan."""
        plan = Plan(
            name=self.name,
            state=self.state.copy(),
            include_state=self.include_state,
            is_built=self.is_built,
            operations=self.operations,
            placeholders=self.placeholders,
            id=sy.ID_PROVIDER.pop(),
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        plan.state.plan = plan

        return plan

    def __setattr__(self, name, value):
        """Add new tensors or parameter attributes to the state and register them
        in the owner's registry
        """
        object.__setattr__(self, name, value)

        if isinstance(value, FrameworkTensor):
            placeholder = sy.PlaceHolder(tags={"#state", f"#{self.var_count + 1}"}, id=value.id)
            self.var_count += 1
            placeholder.instantiate(value)
            self.state.state_placeholders.append(placeholder)
            self.placeholders[value.id] = placeholder
        elif isinstance(value, FrameworkLayerModule):
            for tensor_name, tensor in value.named_tensors():
                self.__setattr__(f"{name}_{tensor_name}", tensor)

    def __call__(self, *args, **kwargs):
        """
        Calls a plan execution with some arguments.

        When possible, run the original function to improve efficiency. When
        it's not, for example if you fetched the plan from a remote worker,
        then run it from the tape of operations:
        - Instantiate input placeholders
        - for each recorded operation, run the operation on the placeholders
          and use the result(s) to instantiate to appropriate placeholder.
        - Return the instantiation of all the output placeholders.
        """
        if self.forward is not None:  # if not self.is_built:
            if self.include_state:
                args = (*args, self.state)
            return self.forward(*args)

        else:
            # Instantiate all the input placeholders in the correct order

            input_placeholders = sorted(self.find_placeholders("#input"), key=tag_sort("input"))
            for placeholder, arg in zip(input_placeholders, args):
                placeholder.instantiate(arg)

            for i, op in enumerate(self.operations):
                cmd, _self, args, kwargs, return_placeholder = (
                    op.cmd_name,
                    op.cmd_owner,  # cmd_owner is equivalent to the "self" in a method
                    op.cmd_args,
                    op.cmd_kwargs,
                    op.return_ids,
                )
                if _self is None:
                    response = eval(cmd)(*args, **kwargs)  # nosec
                else:
                    response = getattr(_self, cmd)(*args, **kwargs)
                return_placeholder.instantiate(response.child)

            # This ensures that we return the output placeholder in the correct order
            output_placeholders = sorted(self.find_placeholders("#output"), key=tag_sort("output"))
            response = [p.child for p in output_placeholders]

            if len(response) == 1:
                return response[0]
            else:
                return tuple(response)

    def run(self, args: Tuple, result_ids: List[Union[str, int]]):
        """Controls local or remote plan execution.
        If the plan doesn't have the plan built, first build it using the original function.

        Args:
            args: Arguments used to run plan.
            result_ids: List of ids where the results will be stored.
        """
        # TODO: can we reuse result_ids?

        # We build the plan only if needed
        if not self.is_built:
            self.build(args)

        result = self.__call__(*args)
        return result

    def send(self, *locations: AbstractWorker, force=False) -> PointerPlan:
        """Send plan to locations.

        If the plan was not built locally it will raise an exception.
        If `force` = true plan is going to be sent either way.

        Args:
            locations: List of workers.
            force: A boolean indicating if this operation should be forced.
        """
        if not self.is_built and not force:
            raise RuntimeError("A plan needs to be built before being sent to a worker.")

        if len(locations) == 1:
            location = locations[0]

            # Check if plan was already sent at the location
            if location in self.pointers:
                return self.pointers[location]

            # Send the Plan
            pointer = self.owner.send(self, workers=location)

            self.pointers[location] = pointer
        else:
            ids_at_location = []
            for location in locations:
                if location in self.pointers:
                    # Use the pointer that was already sent
                    pointer = self.pointers[location]
                else:
                    # Send the Plan
                    pointer = self.owner.send(self, workers=location)

                    self.pointers[location] = pointer

                ids_at_location.append(pointer.id_at_location)

            pointer = sy.PointerPlan(location=locations, id_at_location=ids_at_location)

        return pointer

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
        self, owner, garbage_collect_data, location=None, id_at_location=None, **kwargs
    ):
        return PointerPlan(
            owner=owner,
            location=location or self.owner,
            id_at_location=id_at_location or self.id,
            garbage_collect_data=garbage_collect_data,
        )

    def __str__(self):
        """Returns the string representation of Plan."""
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

        return out

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def simplify(worker: AbstractWorker, plan: "Plan") -> tuple:
        """
        This function takes the attributes of a Plan and saves them in a tuple
        Args:
            worker (AbstractWorker): the worker doing the serialization
            plan (Plan): a Plan object
        Returns:
            tuple: a tuple holding the unique attributes of the Plan object

        """
        return (
            sy.serde.msgpack.serde._simplify(worker, plan.id),
            sy.serde.msgpack.serde._simplify(worker, plan.operations),
            sy.serde.msgpack.serde._simplify(worker, plan.state),
            sy.serde.msgpack.serde._simplify(worker, plan.include_state),
            sy.serde.msgpack.serde._simplify(worker, plan.is_built),
            sy.serde.msgpack.serde._simplify(worker, plan.name),
            sy.serde.msgpack.serde._simplify(worker, plan.tags),
            sy.serde.msgpack.serde._simplify(worker, plan.description),
            sy.serde.msgpack.serde._simplify(worker, plan.placeholders),
        )

    @staticmethod
    def detail(worker: AbstractWorker, plan_tuple: tuple) -> "Plan":
        """This function reconstructs a Plan object given its attributes in the form of a tuple.
        Args:
            worker: the worker doing the deserialization
            plan_tuple: a tuple holding the attributes of the Plan
        Returns:
            plan: a Plan object
        """

        (
            id,
            operations,
            state,
            include_state,
            is_built,
            name,
            tags,
            description,
            placeholders,
        ) = plan_tuple

        worker._tmp_placeholders = {}
        id = sy.serde.msgpack.serde._detail(worker, id)
        operations = sy.serde.msgpack.serde._detail(worker, operations)
        state = sy.serde.msgpack.serde._detail(worker, state)
        placeholders = sy.serde.msgpack.serde._detail(worker, placeholders)

        plan = sy.Plan(
            include_state=include_state,
            is_built=is_built,
            operations=operations,
            placeholders=placeholders,
            id=id,
            owner=worker,
        )
        del worker._tmp_placeholders

        plan.state = state
        state.plan = plan

        plan.name = sy.serde.msgpack.serde._detail(worker, name)
        plan.tags = sy.serde.msgpack.serde._detail(worker, tags)
        plan.description = sy.serde.msgpack.serde._detail(worker, description)

        return plan

    @staticmethod
    def bufferize(worker: AbstractWorker, plan: "Plan") -> PlanPB:
        """
        This function takes the attributes of a Plan and saves them in a Protobuf message
        Args:
            worker (AbstractWorker): the worker doing the serialization
            plan (Plan): a Plan object
        Returns:
            PlanPB: a Protobuf message holding the unique attributes of the Plan object

        """
        protobuf_plan = PlanPB()

        sy.serde.protobuf.proto.set_protobuf_id(protobuf_plan.id, plan.id)

        protobuf_operations = [
            sy.serde.protobuf.serde._bufferize(worker, operation).operation
            for operation in plan.operations
        ]
        protobuf_plan.operations.extend(protobuf_operations)

        protobuf_plan.state.CopyFrom(sy.serde.protobuf.serde._bufferize(worker, plan.state))

        protobuf_plan.include_state = plan.include_state
        protobuf_plan.is_built = plan.is_built
        protobuf_plan.name = plan.name
        protobuf_plan.tags.extend(plan.tags)

        if protobuf_plan.description:
            protobuf_plan.description = plan.description

        if type(plan.placeholders) == type(dict()):
            placeholders = plan.placeholders.values()
        else:
            placeholders = plan.placeholders

        protobuf_placeholders = [
            sy.serde.protobuf.serde._bufferize(worker, placeholder) for placeholder in placeholders
        ]
        protobuf_plan.placeholders.extend(protobuf_placeholders)

        return protobuf_plan

    @staticmethod
    def unbufferize(worker: AbstractWorker, protobuf_plan: PlanPB) -> "Plan":
        """This function reconstructs a Plan object given its attributes in the form of a Protobuf message
        Args:
            worker: the worker doing the deserialization
            protobuf_plan: a Protobuf message holding the attributes of the Plan
        Returns:
            plan: a Plan object
        """

        worker._tmp_placeholders = {}
        id = sy.serde.protobuf.proto.get_protobuf_id(protobuf_plan.id)

        operations = []
        for operation in protobuf_plan.operations:
            op_msg = OperationMessagePB()
            op_msg.operation.CopyFrom(operation)
            operations.append(op_msg)

        operations = [
            sy.serde.protobuf.serde._unbufferize(worker, operation) for operation in operations
        ]
        state = sy.serde.protobuf.serde._unbufferize(worker, protobuf_plan.state)

        placeholders = [
            sy.serde.protobuf.serde._unbufferize(worker, placeholder)
            for placeholder in protobuf_plan.placeholders
        ]
        placeholders = dict([(placeholder.id, placeholder) for placeholder in placeholders])

        plan = sy.Plan(
            include_state=protobuf_plan.include_state,
            is_built=protobuf_plan.is_built,
            operations=operations,
            placeholders=placeholders,
            id=id,
            owner=worker,
        )
        del worker._tmp_placeholders

        plan.state = state
        state.plan = plan

        plan.name = protobuf_plan.name
        if protobuf_plan.tags:
            plan.tags = set(protobuf_plan.tags)
        if protobuf_plan.description:
            plan.description = protobuf_plan.description

        return plan


def tag_sort(keyword):
    """
    Utility function to sort tensors by their (unique) tag including "keyword"
    """
    # TODO is only works up to 9 return values, because comparison is done on str and '7' > '16'
    def extract_key(placeholder):
        for tag in placeholder.tags:
            if keyword in tag:
                return tag

        return TypeError(f"Tag '{keyword}' not found in placeholder tags:", placeholder.tags)

    return extract_key
