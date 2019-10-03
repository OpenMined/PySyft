import copy
import functools
from typing import List
from typing import Tuple
from typing import Union
from typing import Dict

import torch

import syft as sy
from syft.codes import MSGTYPE
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkLayerModule
from syft.generic.object import AbstractObject
from syft.generic.object_storage import ObjectStorage
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker


class func2plan(object):
    """Converts a function to a plan.

    Converts a function containing sequential pytorch code into
    a plan object which can be sent to any arbitrary worker.

    This class should be used only as a decorator.
    """

    def __init__(self, args_shape=None, state=None):
        self.args_shape = args_shape
        self.state_dict = state or {}
        # include_state is used to distinguish if the initial plan a function or a class:
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
            state_dict=self.state_dict,
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


class State(object):
    """The State is a Plan attribute and is used to send tensors along functions.

    It references Plan tensor or parameters attributes using their name, and make
    sure they are provided to remote workers who are sent the Plan.
    """

    def __init__(self, owner, plan=None, keys=None, state_ids=None):
        self.owner = owner
        self.plan = plan
        self.keys = keys or set()
        self.state_ids = state_ids or []

    def __repr__(self):
        return "State: " + ", ".join(self.keys)

    def append(self, key):
        """Insert element to the store by referencing its name. Note that by default, you don't
        need to use it when defining parameters in a model because it is done automatically.

        The method is voluntarily flexible to several inputs
        Example:
            state.append(('key1, 'key2'))
            state.append(['key1, 'key2'])
            state.append('key1, 'key2')

        But it should only be given strings which correspond to plan
        attributes which are of is_valid_type.
        Example:
            'key1' is ok if plan.key1 is a tensor or a parameter
        """
        # If the parent is a function, then state.append is not supported
        if self.plan is not None and self.plan.include_state:
            return ValueError(
                "If you use plans on functions, you should add your state elements in the "
                "decorator directly.\n"
                "Example:\n"
                '@sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([1])})\n'
                "def foo(x, state)\n"
                '\tbias = state.read("bias")\n'
                "\treturn x + bias"
            )

        if isinstance(key, (list, tuple)):
            for k in key:
                self.append(k)
        elif not isinstance(key, str):
            raise ValueError(
                "Don't provide the element to the state but just its name:\n"
                "Example: if you have self.elem1, call self.add_to_state('elem1')."
            )
        else:
            obj = getattr(self.plan, key)
            if isinstance(obj, (FrameworkTensor, FrameworkLayerModule)):
                self.keys.add(key)
            else:
                raise ValueError(
                    f"Obj of type {type(obj)} is not supported in the state.\n"
                    "Use instead tensors or parameters."
                )
        return self

    def __iadd__(self, other_keys):
        return self.append(other_keys)

    def tensors(self) -> List:
        """
        Fetch and return all the state elements.
        Perform a check of consistency on the elements ids.
        """
        tensors = []
        for state_id in self.state_ids:
            tensor = self.owner.get_obj(state_id)
            assert tensor.id == state_id
            tensors.append(tensor)
        return tensors

    def clone_state(self) -> Dict:
        """
        Return a clone of the state elements. Tensor ids are kept.
        """
        return {tensor.id: tensor.clone() for tensor in self.tensors()}

    def copy(self) -> "State":
        state = State(owner=self.owner, keys=self.keys.copy(), state_ids=self.state_ids.copy())
        return state

    def read(self, key):
        """
        Returns one element referenced by the store of keys.
        Note that this only works for plans built locally
        """
        return getattr(self.plan, key)

    def set_(self, state_dict):
        """
        Reset inplace the state by feeding it a dict of tensors or params
        """
        assert list(self.state_ids) == list(state_dict.keys())

        for state_id, new_tensor in state_dict.items():
            tensor = self.owner.get_obj(state_id)

            with torch.no_grad():
                tensor.set_(new_tensor)

            tensor.child = new_tensor.child if new_tensor.is_wrapper else None
            tensor.is_wrapper = new_tensor.is_wrapper
            if tensor.child is None:
                delattr(tensor, "child")

    @staticmethod
    def create_grad_if_missing(tensor):
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is None:
            o = tensor.sum()
            o.backward()
            if tensor.grad is not None:
                tensor.grad -= tensor.grad

    def send_for_build(self, location, **kwargs):
        """
        Send functionality that can only be used when sending the state for
        building the plan. Other than this, you shouldn't need to send the
        state separately.
        """
        assert location.id == self.plan.id  # ensure this is a send for the build

        for tensor in self.tensors():
            self.create_grad_if_missing(tensor)
            tensor.send_(location, **kwargs)

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
    def simplify(state: "State") -> tuple:
        """
        Simplify the plan's state when sending a plan

        Note that keys are not sent, as they are only used for locally
        built plan
        """
        return (sy.serde._simplify(state.state_ids), sy.serde._simplify(state.tensors()))

    @staticmethod
    def detail(worker: AbstractWorker, state_tuple: tuple) -> "State":
        """
        Reconstruct the plan's state from the state elements and supposed
        ids.
        """
        state_ids, state_elements = state_tuple
        state_ids = sy.serde._detail(worker, state_ids)
        state_elements = sy.serde._detail(worker, state_elements)

        for state_id, state_element in zip(state_ids, state_elements):
            worker.register_obj(state_element, obj_id=state_id)

        state = State(owner=worker, plan=None, state_ids=state_ids)
        return state


class Command(object):
    """A command is a serializable object providing instruction to operate designated tensors."""

    def __init__(self, command):
        self.command = command

    def update_ids(self, from_ids, to_ids):
        self.command = Command.replace_ids(self.command, from_ids, to_ids)

    @staticmethod
    def replace_ids(command, from_ids, to_ids):

        assert isinstance(from_ids, (list, tuple))
        assert isinstance(to_ids, (list, tuple))

        type_obj = type(command)
        command = list(command)
        for i, item in enumerate(command):
            if isinstance(item, (int, str, bytes)) and item in from_ids:
                command[i] = to_ids[from_ids.index(item)]
            elif isinstance(item, (list, tuple)):
                command[i] = Command.replace_ids(command=item, from_ids=from_ids, to_ids=to_ids)
        return type_obj(command)


class Procedure(object):
    """A Procedure is a list of commands."""

    def __init__(self, commands=None, arg_ids=None, result_ids=None):
        self.commands = commands or []
        self.arg_ids = arg_ids or []
        self.result_ids = result_ids or []

    def __str__(self):
        return f"<Procedure #commands:{len(self.commands)}>"

    def __repr__(self):
        return self.__str__()

    def update_ids(
        self,
        from_ids: Tuple[Union[str, int]] = [],
        to_ids: Tuple[Union[str, int]] = [],
        from_worker: Union[str, int] = None,
        to_worker: Union[str, int] = None,
    ):
        """Replaces pairs of tensor ids in the plan stored.

        Args:
            from_ids: Ids to change.
            to_ids: Ids to replace with.
            from_worker: The previous worker that built the plan.
            to_worker: The new worker that is running the plan.
        """
        for idx, command in enumerate(self.commands):
            if from_worker and to_worker:
                from_workers, to_workers = [from_worker], [to_worker]
                if isinstance(from_worker, str):
                    from_workers.append(from_worker.encode("utf-8"))
                    to_workers.append(to_worker)
                command = Command.replace_ids(command, from_workers, to_workers)

            if len(from_ids) and len(to_ids):
                command = Command.replace_ids(command, from_ids, to_ids)

            self.commands[idx] = command

        return self

    def update_worker_ids(self, from_worker_id: Union[str, int], to_worker_id: Union[str, int]):
        return self.update_ids([], [], from_worker_id, to_worker_id)

    def update_args(
        self,
        args: Tuple[Union[FrameworkTensorType, AbstractTensor]],
        result_ids: List[Union[str, int]],
    ):
        """Replace args and result_ids with the ones given.
        Updates the arguments ids and result ids used to execute
        the plan.
        Args:
            args: List of tensors.
            result_ids: Ids where the plan output will be stored.
        """

        arg_ids = tuple(arg.id for arg in args)
        self.update_ids(self.arg_ids, arg_ids)
        self.arg_ids = arg_ids

        self.update_ids(self.result_ids, result_ids)
        self.result_ids = result_ids

    def copy(self) -> "Procedure":
        procedure = Procedure(
            commands=copy.deepcopy(self.commands), arg_ids=self.arg_ids, result_ids=self.result_ids
        )
        return procedure

    @staticmethod
    def simplify(procedure: "Procedure") -> tuple:
        return (
            tuple(
                procedure.commands
            ),  # We're not simplifying because commands are already simplified
            sy.serde._simplify(procedure.arg_ids),
            sy.serde._simplify(procedure.result_ids),
        )

    @staticmethod
    def detail(worker: AbstractWorker, procedure_tuple: tuple) -> "State":
        commands, arg_ids, result_ids = procedure_tuple
        commands = list(commands)
        arg_ids = sy.serde._detail(worker, arg_ids)
        result_ids = sy.serde._detail(worker, result_ids)

        procedure = Procedure(commands, arg_ids, result_ids)
        return procedure


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
        state_ids: ids of the state elements
        arg_ids: ids of the last argument ids (present in the procedure commands)
        result_ids: ids of the last result ids (present in the procedure commands)
        readable_plan: list of commands
        blueprint: the function to be transformed into a plan
        state_dict: a dict of state elements whose keys will be used to transform them in attributes. It can be used to populate a state
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
        state_ids: List[Union[str, int]] = None,
        arg_ids: List[Union[str, int]] = None,
        result_ids: List[Union[str, int]] = None,
        readable_plan: List = None,
        blueprint=None,
        state_dict=None,
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

        # Info about the plan stored via the state and the procedure
        self.procedure = procedure or Procedure(readable_plan, arg_ids, result_ids)
        self.state = state or State(owner=owner, plan=self, state_ids=state_ids)
        if state_dict is not None:
            for key, tensor in state_dict.items():
                setattr(self, key, tensor)
        self.include_state = include_state
        self.is_built = is_built

        if blueprint is not None:
            self.forward = blueprint
        elif self.is_built:
            self.forward = None

    def add_to_state(self, *elements):
        for elem in elements:
            self.state.append(elem)

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
        return self.procedure.commands

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

    def _recv_msg(self, bin_message: bin):
        """Upon reception, a Plan stores all commands which can be executed lazily.
        Args:
            bin_message: the message of a command received.
        Returns:
            The None message serialized to specify the command was received.
        """
        (some_type, (msg_type, contents)) = sy.serde.deserialize(bin_message, details=False)

        if (
            msg_type not in (MSGTYPE.OBJ, MSGTYPE.OBJ_DEL, MSGTYPE.FORCE_OBJ_DEL)
            and not self.is_built
        ):
            self.procedure.commands.append((some_type, (msg_type, contents)))

        # we can't receive the results of a plan without
        # executing it. So, execute the plan.
        if msg_type in (MSGTYPE.OBJ_REQ, MSGTYPE.IS_NONE, MSGTYPE.GET_SHAPE):
            return self.__call__()

        return sy.serde.serialize(None)

    def build(self, *args):
        """Builds the plan.

        The build operation is done "on" the plan (which can be seen like a
        worker), by running the forward function. The plan will therefore
        execute each command of the function, will cache it using _recv_msg
        and will use it to fill the plan.
        To do so, all the arguments provided and the state elements should be
        moved to the plan, using send().

        Args:
            args: Input data.
        """

        # Move the arguments of the first call to the plan and store their ids
        # as they will be included in the commands: it should be updated
        # when the function is called with new args and that's why we keep the
        # refs self.procedure.arg_ids
        arg_ids = list()
        build_args = list()
        for arg in args:
            arg = arg.send(self)
            arg_ids.append(arg.id_at_location)
            build_args.append(arg)
        self.procedure.arg_ids = tuple(arg_ids)

        # Same for the state element: we send them to the plan and keep reference
        # to the remote ids
        cloned_state = self.state.clone_state()
        build_state_ids = tuple(self.state.state_ids)  # tuple makes a copy
        self.state.send_for_build(location=self)

        # We usually have include_state==True for functions converted to plan
        # using @func2plan and we need therefore to add the state manually
        if self.include_state:
            res_ptr = self.forward(*build_args, self.state)
        else:
            res_ptr = self.forward(*build_args)

        # We put back a clone of the original state
        self.state.set_(cloned_state)
        # and update the procedure
        self.procedure.update_ids(from_ids=build_state_ids, to_ids=self.state.state_ids)

        # The plan is now built, we hide the fact that it was run on
        # the plan and not on the owner by replacing the workers ids
        self.procedure.update_worker_ids(from_worker_id=self.id, to_worker_id=self.owner.id)

        # The id where the result should be stored
        self.procedure.result_ids = (res_ptr.id_at_location,)

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

        plan.state.set_(self.state.clone_state())

        # Replace occurences of the old id to the new plan id
        plan.procedure.update_worker_ids(self.id, plan.id)

        return plan

    def __setattr__(self, name, value):
        """Add new tensors or parameter attributes to the state and register them
        in the owner's registry
        """
        object.__setattr__(self, name, value)

        if isinstance(value, FrameworkTensor):
            self.state.state_ids.append(value.id)
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
        if len(kwargs):
            raise ValueError("Kwargs are not supported for plan.")

        result_ids = [sy.ID_PROVIDER.pop()]

        if self.forward is not None:
            if self.include_state:
                args = (*args, self.state)
            return self.forward(*args)
        else:
            return self.run(args, result_ids=result_ids)

    def execute_commands(self):
        for message in self.procedure.commands:
            bin_message = sy.serde.serialize(message, simplified=True)
            _ = self.owner.recv_msg(bin_message)

    def run(self, args: Tuple, result_ids: List[Union[str, int]]):
        """Controls local or remote plan execution.

        If the plan doesn't have the plan built, first build it using the blueprint.
        Then, update the plan with the result_ids and args ids given, run the plan
        commands, build pointer(s) to the response(s) and return.

        Args:
            args: Arguments used to run plan.
            result_ids: List of ids where the results will be stored.
        """
        # We build the plan only if needed
        if not self.is_built:
            self.build(args)

        self.procedure.update_args(args, result_ids)

        self.execute_commands()
        responses = [self.owner.get_obj(result_id) for result_id in result_ids]

        if len(responses) == 1:
            return responses[0]
        return responses

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

            self.procedure.update_worker_ids(self.owner.id, location.id)
            # Send the Plan
            pointer = self.owner.send(self, workers=location)
            # Revert ids
            self.procedure.update_worker_ids(location.id, self.owner.id)
        else:
            ids_at_location = []
            for location in locations:
                self.procedure.update_worker_ids(self.owner.id, location.id)
                # Send the Plan
                pointer = self.owner.send(self, workers=location)
                # Revert ids
                self.procedure.update_worker_ids(location.id, self.owner.id)
                ids_at_location.append(pointer.id_at_location)

            pointer = sy.PointerPlan(location=locations, id_at_location=ids_at_location)

        return pointer

    def get_(self):
        self.state.get_()
        return self

    get = get_

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
    def simplify(plan: "Plan") -> tuple:
        """
        This function takes the attributes of a Plan and saves them in a tuple
        Args:
            plan (Plan): a Plan object
        Returns:
            tuple: a tuple holding the unique attributes of the Plan object

        """
        return (
            sy.serde._simplify(plan.id),
            sy.serde._simplify(plan.procedure),
            sy.serde._simplify(plan.state),
            sy.serde._simplify(plan.include_state),
            sy.serde._simplify(plan.is_built),
            sy.serde._simplify(plan.name),
            sy.serde._simplify(plan.tags),
            sy.serde._simplify(plan.description),
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

        id, procedure, state, include_state, is_built, name, tags, description = plan_tuple
        id = sy.serde._detail(worker, id)
        procedure = sy.serde._detail(worker, procedure)
        state = sy.serde._detail(worker, state)

        plan = sy.Plan(owner=worker, id=id, include_state=include_state, is_built=is_built)

        plan.procedure = procedure
        plan.state = state
        state.plan = plan

        plan.name = sy.serde._detail(worker, name)
        plan.tags = sy.serde._detail(worker, tags)
        plan.description = sy.serde._detail(worker, description)

        return plan
