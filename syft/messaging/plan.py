import copy
import functools
from typing import List
from typing import Tuple
from typing import Union
from typing import Set

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

    def __init__(self, plan, keys=None, state_ids=None, state_dict=None):
        self.plan = plan
        self.keys = keys or set()
        self.state_ids = state_ids or []
        if state_dict is not None:
            self.set_(state_dict)

    def __repr__(self):
        return "State: " + ", ".join(self.keys)

    def append(self, key):
        """Insert element to the store by referencing its name.

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
        if isinstance(key, (list, tuple)):
            for k in key:
                self.append(k)
        elif not isinstance(key, str):
            raise ValueError(
                "Don't provide the element to the state but just its name:\n"
                "Example: if you have self.elem1, call self.add_to_state('elem1')."
            )
        else:
            t = getattr(self.plan, key)
            if self.is_valid_type(t):
                self.keys.add(key)
            else:
                raise ValueError(
                    f"Obj of type {type(t)} is not supported in the state.\n"
                    "Use instead tensors or parameters."
                )
        return self

    def __iadd__(self, other_keys):
        return self.append(other_keys)

    def is_valid_type(self, obj):
        return isinstance(obj, (torch.nn.Module, torch.Tensor))

    def get_ids_at_location(self):
        """
        Returns all the id_at_location of the pointers in the state
        """
        ids_at_location = []

        for key in self.keys:
            ptr = getattr(self.plan, key)
            if isinstance(ptr, torch.nn.Module):
                for param in ptr.parameters():
                    ids_at_location.append(param.id_at_location)
            else:
                ids_at_location.append(ptr.id_at_location)

        return ids_at_location

    def copy_state(self):
        copied_state = {}
        for key in self.keys:
            t = getattr(self.plan, key)
            copied_state[key] = t.copy()
        return copied_state

    def copy(self) -> "State":
        state = State(plan=None, keys=self.keys.copy(), state_ids=self.state_ids.copy())
        return state

    def read(self, key):
        """
        Returns one element referenced by the store
        """
        return getattr(self.plan, key)

    def set_(self, state_dict):
        """
        Reset inplace the state by feeding it a dict of tensors or params
        """
        for key in self.keys:
            if hasattr(self.plan, key):
                delattr(self.plan, key)

        for key in state_dict.keys():
            setattr(self.plan, key, state_dict[key])

        self.keys = list(state_dict.keys())

    def is_missing_grad(self, t):
        return isinstance(t, torch.nn.Parameter) and t.grad is None

    def create_grad_objects(self, p):
        o = p.sum()
        o.backward()
        if p.grad is not None:
            p.grad -= p.grad

    def send(self, location, **kwargs):
        for key in self.keys:
            t = getattr(self.plan, key)

            if self.is_missing_grad(t):
                self.create_grad_objects(t)

            t.send_(location, **kwargs)

    def get(self):
        for key in self.keys:
            t = getattr(self.plan, key)
            t.get_()

    def fix_precision_(self, *args, **kwargs):
        for key in self.keys:
            t = getattr(self.plan, key)

            if self.is_missing_grad(t):
                self.create_grad_objects(t)

            setattr(self.plan, key, t.fix_precision(*args, **kwargs))

    def float_precision_(self):
        for key in self.keys:
            t = getattr(self.plan, key)
            t.float_precision_()

    def share_(self, *args, **kwargs):
        for key in self.keys:
            t = getattr(self.plan, key)

            if self.is_missing_grad(t):
                self.create_grad_objects(t)

            t.share_(*args, **kwargs)

    @staticmethod
    def simplify(state: "State") -> tuple:
        return (
            sy.serde._simplify(state.keys),
            sy.serde._simplify(state.state_ids),
            sy.serde._simplify([getattr(state.plan, key) for key in state.keys]),
        )

    @staticmethod
    def detail(worker: AbstractWorker, state_tuple: tuple) -> "State":
        keys, state_ids, state_elements = state_tuple
        keys = sy.serde._detail(worker, keys)
        state_ids = sy.serde._detail(worker, state_ids)
        state_elements = sy.serde._detail(worker, state_elements)
        assert isinstance(state_elements, list)
        for state_id, state_element in zip(state_ids, state_elements):
            worker.register_obj(state_element, obj_id=state_id)

        state = State(plan=None)
        state.keys = keys
        state.state_ids = state_ids

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


class Plan(AbstractObject, ObjectStorage, torch.nn.Module):
    """A Plan stores a sequence of torch operations, just like a function.

    A Plan is intended to store a sequence of torch operations, just like a function,
    but it allows to send this sequence of operations to remote workers and to keep a
    reference to it. This way, to compute remotely this sequence of operations on some remote
    input referenced through pointers, instead of sending multiple messages you need now to send a
    single message with the references of the plan and the pointers.
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
        torch.nn.Module.__init__(self)

        # Plan instance info
        self.name = name or self.__class__.__name__
        self._owner = owner

        # Info about the plan stored vua the state and the procedure
        self.procedure = procedure or Procedure(readable_plan, arg_ids, result_ids)
        self.state = state or State(self, state_dict=state_dict, state_ids=state_ids)
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

    # Override property of nn.Module
    @property
    def owner(self):
        return self._owner

    @owner.setter
    def owner(self, new_owner):
        self._owner = new_owner

    # Override property of nn.Module
    @property
    def location(self):
        raise AttributeError("Plan has no attribute location")

    # For backward compatibility
    @property
    def readable_plan(self):
        return self.procedure.commands

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
        # refs self.arg_ids
        arg_ids = list()
        build_args = list()
        for arg in args:
            arg = arg.send(self)
            arg_ids.append(arg.id_at_location)
            build_args.append(arg)
        self.procedure.arg_ids = tuple(arg_ids)

        # Same for the state element: we send them to the plan and keep reference
        # to the ids
        copied_state_dict = self.state.copy_state()
        self.state.send(location=self)
        self.state.state_ids = self.state.get_ids_at_location()  # todo is it needed?

        # We usually have include_state==True for functions converted to plan
        # using @func2plan and we need therefore to add the state manually
        if self.include_state:
            res_ptr = self.forward(*build_args, self.state)
        else:
            res_ptr = self.forward(*build_args)
        res_ptr.child.garbage_collect_data = False

        # We put back the original state. There is no need to keep a reference
        # To the element sent to the plan.
        self.state.set_(copied_state_dict)

        # The readable plan is now built, we hide the fact that it was run on
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

        plan.state.set_(self.state.copy_state())

        # Replace occurences of the old id to the new plan id
        plan.procedure.update_worker_ids(self.id, plan.id)

        return plan

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

    # def build_pointer_responses(self, result_ids, return_ptr=False):
    #     responses = []
    #     for return_id in result_ids:
    #         response = PointerTensor(
    #             location=self.owner, id_at_location=return_id, owner=self, id=sy.ID_PROVIDER.pop()
    #         )
    #         responses.append(response if return_ptr else response.get())
    #
    #     return responses

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

    def send(self, *locations, force=False):
        """Send plan to locations.

        If the plan was not built locally it will raise an exception.
        If `force` = true plan is going to be sent either way.

        Args:
            locations: List of workers.
            force: A boolean indicating if this operation should be forced.
        """
        if not self.is_built and not force:
            raise RuntimeError("A plan needs to be built before being sent to a worker.")

        if len(locations) != 1:
            raise NotImplementedError(
                "Plans can't be sent to multiple locations. Waiting for MultiPointerPlan."
            )
        location = locations[0]

        # Might be useless now
        # state_ptr_ids = remote_state.get_ids_at_location()
        # remote_commands.update_ids(self.state.state_ids, state_ptr_ids, self.owner.id, location.id)

        self.procedure.update_worker_ids(self.owner.id, location.id)

        # Send the Plan
        pointer = self.owner.send(self, workers=location)

        self.procedure.update_worker_ids(location.id, self.owner.id)

        return pointer

    def get(self):
        self.state.get()
        return self
        # TODO add the following in a smart way?
        # raise NotImplementedError("You should call .get() on a PointerPlan, not on a Plan.")

    def fix_precision_(self, *args, **kwargs):
        self.state.fix_precision_(*args, **kwargs)
        return self

    fix_precision = fix_precision_
    fix_prec = fix_precision_

    def float_precision_(self):
        self.state.float_precision_()
        return self

    float_precision = float_precision_
    float_prec = float_precision_

    def share_(self, *args, **kwargs):
        self.state.share_(*args, **kwargs)
        return self

    share = share_

    def create_pointer(self, owner, location, id_at_location, garbage_collect_data, **kwargs):
        return PointerPlan(
            owner=owner,
            location=location,
            id_at_location=id_at_location,
            garbage_collect_data=garbage_collect_data,
        )

    def __str__(self):
        """Returns the string representation of PlanWorker.
        Note:
            __repr__ calls this method by default.
        """
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

        for key, state_id in zip(state.keys, state.state_ids):
            setattr(plan, key, worker.get_obj(state_id))

        plan.procedure = procedure
        plan.state = state
        state.plan = plan

        plan.name = sy.serde._detail(worker, name)
        plan.tags = sy.serde._detail(worker, tags)
        plan.description = sy.serde._detail(worker, description)

        return plan
