from typing import List
from typing import Tuple
from typing import Union

import syft as sy
import torch
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkLayerModule
from syft.generic.object import AbstractObject
from syft.generic.object_storage import ObjectStorage
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.messaging.message import Operation
from syft.messaging.plan.procedure import Procedure
from syft.messaging.plan.state import State
from syft.workers.abstract import AbstractWorker
from syft.frameworks.torch.tensors.interpreters.promise import PromiseTensor
from syft.frameworks.torch.tensors.interpreters.placeholder import PlaceHolder


class func2plan(object):
    """Converts a function to a plan.

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

    def __call__(self, plan_blueprint):
        plan = Plan(
            owner=sy.local_worker,
            id=sy.ID_PROVIDER.pop(),
            name=plan_blueprint.__name__,
            blueprint=plan_blueprint,
            state_tensors=self.state_tensors,
            include_state=self.include_state,
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
        procedure: stores and manages the plan's commands
        state: store the plan tensors like model parameters
        include_state: if true, implies that the plan is a function, else a class. If true, the
            state is re-integrated in the args to be accessed within the function
        is_built: state if the plan has already been built.
        state_placeholders: ids of the state elements
        arg_placeholders: ids of the last argument ids (present in the procedure commands)
        result_placeholders: ids of the last result ids (present in the procedure commands)
        readable_plan: list of commands
        blueprint: the function to be transformed into a plan
        state_tensors: a tuple of state elements. It can be used to populate a state
        id: state id
        owner: state owner
        tags: state tags
        description: state decription
    """

    def __init__(
        self,
        name: str = None,
        procedure: Procedure = None,
        state: State = None,
        include_state: bool = False,
        is_built: bool = False,
        # Optional kwargs if commands or state are not provided
        state_placeholders: List[PlaceHolder] = None,
        input_placeholders: List[PlaceHolder] = None,
        output_placeholders: List[PlaceHolder] = None,
        readable_plan: List = None,
        blueprint=None,
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

        # If we have plans in plans we need to keep track of the states for each plan
        # because we will need to serialize and send them to the remote workers
        self.nested_states = []

        # Info about the plan stored via the state and the procedure
        self.procedure = procedure or Procedure(readable_plan, input_placeholders, output_placeholders)
        self.state = state or State(owner=owner, plan=self, state_placeholders=state_placeholders)
        if state_tensors is not None:
            for tensor in state_tensors:
                placeholder = sy.PlaceHolder()
                placeholder.instanciate(tensor)
                self.state.state_placeholders.append(placeholder)
                self.owner.register_obj(tensor)

        self.include_state = include_state
        self.is_built = is_built

        self.input_placeholders = []
        self.placeholders = {}

        # The plan has not been sent
        self.pointers = dict()

        if blueprint is not None:
            self.forward = blueprint
        elif self.is_built:
            self.forward = None

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
        return self.procedure.operations

    def parameters(self):
        """
        This is defined to match the torch api of nn.Module where .parameters() return the model tensors / parameters
        """
        return self.state.tensors()

    @property
    def output_shape(self):
        if self._output_shape is None:
            args = self._create_placeholders(self.input_shapes)
            # NOTE I currently need to regiser and then remove objects to use the method
            # but a better syntax is being worked on
            for arg in args:
                self.owner.register_obj(arg)
            output = self(*args)
            for arg in args:
                self.owner.rm_obj(arg)
            self._output_shape = output.shape
        return self._output_shape

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

    def add_placeholder(self, tensor, find_inputs=False):
        if tensor.id not in self.placeholders.keys():
            placeholder = sy.PlaceHolder(tags={f'#{self.var_count + 1}'})
            self.placeholders[tensor.id] = placeholder
            if find_inputs:
                self.input_placeholders.append(placeholder)
                placeholder.tags.add('#input')
            self.var_count += 1

        return self.placeholders[tensor.id]

    def replace_with_placeholders(self, obj, **kw):
        if isinstance(obj, (tuple, list)):
            r = [self.replace_with_placeholders(o, **kw) for o in obj]
            return type(obj)(r)
        elif isinstance(obj, dict):
            return {
                key: self.replace_with_placeholders(value, **kw)
                for key, value in obj.items()
            }
        elif isinstance(obj, torch.Tensor):
            return self.add_placeholder(obj, **kw)
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif obj is None:
            return None
        else:
            raise TypeError(f"Type {type(obj)} not supported in plans args/kwargs")

    def build(self, *args):
        """Builds the plan.

        Args:
            args: Input data.
        """

        # Same for the state element: we keep a clone
        cloned_state = self.state.clone_state_dict()

        self.owner.init_plan = self
        sy.hook.trace, sy.hook.trace_inactive = True, True

        self.var_count = 0

        # We usually have include_state==True for functions converted to plan
        # using @func2plan and we need therefore to add the state manually
        if self.include_state:
            res = self.forward(*args, self.state)
        else:
            res = self.forward(*args)

        sy.hook.trace, sy.hook.trace_inactive = False, False
        self.owner.init_plan = None

        # We put back the clone of the original state
        self.state.set_(cloned_state)

        for log in sy.trace_logs:
            req, resp = log
            req, resp = self.replace_with_placeholders(req, find_inputs=True), self.replace_with_placeholders(resp)
            operation = (req, resp)
            self.procedure.operations.append(operation)

        sy.trace_logs = []

        self.is_built = True

    def copy(self):
        """Creates a copy of a plan."""
        plan = Plan(
            name=self.name,
            procedure=self.procedure.copy(),
            state=self.state.copy(),
            include_state=self.include_state,
            is_built=self.is_built,
            # General kwargs
            id=sy.ID_PROVIDER.pop(),
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        plan.state.plan = plan

        plan.state.set_(self.state.clone_state_dict())

        # Replace occurences of the old id to the new plan id
        plan.procedure.update_worker_ids(self.id, plan.id)

        return plan

    def __setattr__(self, name, value):
        """Add new tensors or parameter attributes to the state and register them
        in the owner's registry
        """
        object.__setattr__(self, name, value)

        if isinstance(value, FrameworkTensor):
            self.state.state_placeholders.append(value.id)
            self.owner.register_obj(value)
        elif isinstance(value, FrameworkLayerModule):
            for tensor_name, tensor in value.named_tensors():
                self.__setattr__(f"{name}_{tensor_name}", tensor)

    def __call__(self, *args, **kwargs):
        """Calls a plan.

        Calls a plan execution with some arguments, and specify the ids where the result
        should be stored.

        Returns:
            The pointer to the result of the execution if the plan was already sent,
            else the None message serialized.
        """

        # result_ids = [sy.ID_PROVIDER.pop()]
        #
        # res = None
        # if self.forward is not None:
        #     if self.include_state:
        #         args = (*args, self.state)
        #     res = self.forward(*args)
        # else:
        #     res = self.run(args, result_ids=result_ids)
        #

        print('\nInstanciating inputs..\n')
        # Simulate instanciation
        for placeholder, arg in zip(self.input_placeholders, args):
            placeholder.instanciate(arg)

        print('Running operations...\n')
        for i, operation in enumerate(self.procedure.operations):
            print('run cmd', i)
            (cmd, self, args, kwargs), resp = operation
            print((cmd, self, args, kwargs))
            if self is None:
                r = eval(cmd)(*args, **kwargs)
            else:
                r = getattr(self, cmd)(*args, **kwargs)
            resp.instanciate(r.child)
            print(resp)

        return resp.child

    def send(self, *locations, force=False) -> PointerPlan:
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
            # Revert ids
            self.procedure.update_worker_ids(location.id, self.owner.id)

            self.pointers[location] = pointer
        else:
            ids_at_location = []
            for location in locations:
                if location in self.pointers:
                    # Use the pointer that was already sent
                    pointer = self.pointers[location]
                else:
                    self.procedure.update_worker_ids(self.owner.id, location.id)
                    # Send the Plan
                    pointer = self.owner.send(self, workers=location)
                    # Revert ids
                    self.procedure.update_worker_ids(location.id, self.owner.id)
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
            sy.serde.msgpack.serde._simplify(worker, plan.procedure),
            sy.serde.msgpack.serde._simplify(worker, plan.state),
            sy.serde.msgpack.serde._simplify(worker, plan.include_state),
            sy.serde.msgpack.serde._simplify(worker, plan.is_built),
            sy.serde.msgpack.serde._simplify(worker, plan._output_shape),
            sy.serde.msgpack.serde._simplify(worker, plan.name),
            sy.serde.msgpack.serde._simplify(worker, plan.tags),
            sy.serde.msgpack.serde._simplify(worker, plan.description),
            sy.serde.msgpack.serde._simplify(worker, plan.nested_states),
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
            procedure,
            state,
            include_state,
            is_built,
            input_shapes,
            output_shape,
            name,
            tags,
            description,
            nested_states,
        ) = plan_tuple
        id = sy.serde.msgpack.serde._detail(worker, id)
        procedure = sy.serde.msgpack.serde._detail(worker, procedure)
        state = sy.serde.msgpack.serde._detail(worker, state)
        input_shapes = sy.serde.msgpack.serde._detail(worker, input_shapes)
        output_shape = sy.serde.msgpack.serde._detail(worker, output_shape)
        nested_states = sy.serde.msgpack.serde._detail(worker, nested_states)

        plan = sy.Plan(owner=worker, id=id, include_state=include_state, is_built=is_built)

        plan.nested_states = nested_states
        plan.procedure = procedure
        plan.state = state
        state.plan = plan
        plan.input_shapes = input_shapes
        plan._output_shape = output_shape

        plan.name = sy.serde.msgpack.serde._detail(worker, name)
        plan.tags = sy.serde.msgpack.serde._detail(worker, tags)
        plan.description = sy.serde.msgpack.serde._detail(worker, description)

        return plan
