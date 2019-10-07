import copy
import functools
from typing import List
from typing import Tuple
from typing import Union

import torch

import syft as sy
from syft.codes import MSGTYPE
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.frameworks.types import FrameworkTensorType
from syft.generic.object_storage import ObjectStorage
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.generic.tensor import AbstractTensor
from syft.workers.abstract import AbstractWorker


def make_plan(plan_blueprint):
    """Creates a plan from a function.

    For folks who would prefer to not use a decorator, they can use this function
    to create a plan.
    """
    return func2plan(plan_blueprint)


class func2plan(object):
    """Converts a function to a plan.

    Converts a function containing sequential pytorch code into
    a plan object which can be sent to any arbitrary worker.

    This class should be used only as decorator.
    """

    def __init__(self, args_shape=None, state=None, verbose=False):
        self.args_shape = args_shape
        self.verbose = verbose
        if state is not None:
            self.state = state
            self.include_state = True
        else:
            self.state = {}
            self.include_state = False

    def __call__(self, plan_blueprint):
        plan = Plan(
            owner=sy.local_worker,
            id=sy.ID_PROVIDER.pop(),
            name=plan_blueprint.__name__,
            blueprint=plan_blueprint,
            state=self.state,
            include_state=self.include_state,
            verbose=self.verbose,
        )
        if self.args_shape:
            plan._auto_build(args_shape=self.args_shape)
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

    def __init__(self, plan):
        self.keys = set()
        self.plan = plan

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

    def get_id_at_location(self):
        """
        Returns all the id_at_location of the pointers in the state
        """
        id_at_location = []

        for key in self.keys:
            ptr = getattr(self.plan, key)
            if isinstance(ptr, torch.nn.Module):
                for param in ptr.parameters():
                    id_at_location.append(param.id_at_location)
            else:
                id_at_location.append(ptr.id_at_location)

        return id_at_location

    def copy(self):
        copied_state = {}
        for key in self.keys:
            t = getattr(self.plan, key)
            copied_state[key] = t.copy()
        return copied_state

    def read(self, key):
        """
        Returns one element referenced by the store
        """
        return getattr(self.plan, key)

    def set_(self, dict_state):
        """
        Reset inplace the state by feeding it a dict of tensors or params
        """
        for key in self.keys:
            delattr(self.plan, key)

        for key in dict_state.keys():
            setattr(self.plan, key, dict_state[key])

        self.keys = list(dict_state.keys())

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


class Plan(ObjectStorage, torch.nn.Module):
    """A Plan stores a sequence of torch operations, just like a function.

    A Plan is intended to store a sequence of torch operations, just like a function,
    but it allows to send this sequence of operations to remote workers and to keep a
    reference to it. This way, to compute remotely this sequence of operations on some remote
    input referenced through pointers, instead of sending multiple messages you need now to send a
    single message with the references of the plan and the pointers.
    """

    def __init__(
        self,
        id: Union[str, int] = None,
        owner: "sy.workers.BaseWorker" = None,
        name: str = "",
        state_ids: List[Union[str, int]] = None,
        arg_ids: List[Union[str, int]] = None,
        result_ids: List[Union[str, int]] = None,
        readable_plan: List = None,
        blueprint=None,
        state=None,
        include_state: bool = False,
        is_built: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        ObjectStorage.__init__(self)
        torch.nn.Module.__init__(self)

        # Plan instance info
        self.id = sy.ID_PROVIDER.pop() if id is None else id
        self.name = self.__class__.__name__ if name == "" else name
        self._owner = sy.local_worker if owner is None else owner
        self.verbose = verbose

        # Info about the plan stored
        self.plan = list()
        self.readable_plan = readable_plan if readable_plan is not None else []
        self.state = State(self)
        self.include_state = include_state
        if state is not None:
            self.state.set_(state)
        self.state_ids = state_ids if state_ids is not None else []
        self.arg_ids = arg_ids if arg_ids is not None else []
        self.result_ids = result_ids if result_ids is not None else []
        self.owner_when_built = None
        self.is_built = is_built

        # Pointing info towards a remote plan
        self.locations = []
        self.ptr_plans = {}

        self.tags = None
        self.description = None

        if blueprint is not None:
            self.forward = blueprint
        elif self.is_built:
            self.forward = None

    def add_to_state(self, *elements):
        for elem in elements:
            self.state.append(elem)

    def _auto_build(self, args_shape: List[Tuple[int]] = None):
        args = self._create_placeholders(args_shape)
        self.build(*args)

    def _create_placeholders(self, args_shape):
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
    def location(nn_self):
        raise AttributeError("Plan has no attribute location")

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

        if self.verbose:
            print(f"worker {self} received {sy.codes.code2MSGTYPE[msg_type]} {contents}")

        if (
            msg_type not in (MSGTYPE.OBJ, MSGTYPE.OBJ_DEL, MSGTYPE.FORCE_OBJ_DEL)
            and not self.is_built
        ):
            self.plan.append(bin_message)
            self.readable_plan.append((some_type, (msg_type, contents)))

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
        args = list(args)

        # Move the arguments of the first call to the plan and store their ids
        # as they will be included in the readable_plan: it should be updated
        # when the function is called with new args and that's why we keep the
        # refs self.arg_ids
        self.arg_ids = list()
        local_args = list()
        for arg in args:
            arg = arg.send(self)
            self.arg_ids.append(arg.id_at_location)
            local_args.append(arg)

        # Same for the state element: we send them to the plan and keep reference
        # to the ids
        copied_state = self.state.copy()
        self.state.send(location=self)
        self.state_ids = self.state.get_id_at_location()

        # We usually have include_state==True for functions converted to plan
        # using @func2plan and we need therefore to add the state manually
        if self.include_state:
            res_ptr = self.forward(*local_args, self.state)
        else:
            res_ptr = self.forward(*local_args)
        res_ptr.child.garbage_collect_data = False

        # We put back the original state. There is no need to keep a reference
        # To the element sent to the plan.
        self.state.set_(copied_state)

        # The readable plan is now built, we hide the fact that it was run on
        # the plan and not on the owner by replacing the workers ids
        worker = self.find_location(local_args)
        self.replace_worker_ids(worker.id, self.owner.id)

        # The id where the result should be stored
        self.result_ids = tuple([res_ptr.id_at_location])

        # Store owner that built the plan
        self.owner_when_built = self.owner

        self.is_built = True

    def find_location(self, args):
        """
        Return location if args contain pointers else the local worker
        """
        for arg in args:
            if isinstance(arg, FrameworkTensor):
                if hasattr(arg, "child") and isinstance(arg.child, PointerTensor):
                    return arg.location
        return sy.framework.hook.local_worker

    def copy(self):
        """Creates a copy of a plan."""
        plan = Plan(
            sy.ID_PROVIDER.pop(),
            self.owner,
            self.name,
            arg_ids=self.arg_ids,
            result_ids=self.result_ids,
            readable_plan=self.readable_plan,
            is_built=self.is_built,
        )

        # TODO: we shouldn't use the same state_ids since we want to avoid
        # strange behaviours such as:
        #
        # @sy.func2plan(args_shape=[(1,)], state={"bias": th.tensor([3.0])})
        # def plan(data, state):
        #   bias = state.read("bias")
        #   return data * bias
        #
        # assert plan(th.tensor(1.)) == th.tensor(3.)  # True
        # plan_copy = plan.copy()
        # plan_copy.bias = th.tensor([4.0])
        # assert plan(th.tensor(1.)) == th.tensor(3.)  # False, OMG!!!
        # Issue: https://github.com/OpenMined/PySyft/issues/2601

        plan.state_ids = self.state_ids

        # Replace occurences of the old id to the new plan id
        plan.replace_worker_ids(self.id, plan.id)

        return plan

    def replace_ids(
        self,
        from_ids: List[Union[str, int]],
        to_ids: List[Union[str, int]],
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
        if from_worker is None:
            from_worker = self.id
        if to_worker is None:
            to_worker = self.owner.id

        self.readable_plan = list(self.readable_plan)

        # for every pair of id
        for i in range(len(from_ids)):
            # for every message of the plan
            for j, msg in enumerate(self.readable_plan):
                # look for the old id and replace it with the new one
                self.readable_plan[j] = Plan._replace_message_ids(
                    obj=msg,
                    change_id=from_ids[i],
                    to_id=to_ids[i],
                    from_worker=from_worker,
                    to_worker=to_worker,
                )

        return self

    def replace_worker_ids(self, from_worker_id: Union[str, int], to_worker_id: Union[str, int]):
        """
        Replace occurrences of from_worker_id by to_worker_id in the plan stored
        Works also if those ids are encoded in bytes (for string).

        Args:
            from_worker_id: Id of the the replaced worker.
            to_worker_id: Id of the new worker.
        """

        id_pairs = [(from_worker_id, to_worker_id)]
        if type(from_worker_id) == str:
            to_worker_id_encoded = (
                to_worker_id.encode() if type(to_worker_id) == str else to_worker_id
            )
            id_pairs.append((from_worker_id.encode(), to_worker_id_encoded))

        for id_pair in id_pairs:
            self.readable_plan = Plan._replace_message_ids(
                obj=self.readable_plan,
                change_id=-1,
                to_id=-1,
                from_worker=id_pair[0],
                to_worker=id_pair[1],
            )

    @staticmethod
    def _replace_message_ids(obj, change_id, to_id, from_worker, to_worker):
        _obj = list()

        for item in obj:
            if isinstance(item, int) and (item == change_id):
                _obj.append(to_id)

            elif isinstance(item, type(from_worker)) and (item == from_worker):
                _obj.append(to_worker)

            elif isinstance(item, (list, tuple)):
                _obj.append(
                    Plan._replace_message_ids(
                        obj=item,
                        change_id=change_id,
                        to_id=to_id,
                        from_worker=from_worker,
                        to_worker=to_worker,
                    )
                )

            else:
                _obj.append(item)

        return tuple(_obj)

    def _execute_readable_plan(self, *args):
        # TODO: for now only one value is returned from a plan
        result_ids = [sy.ID_PROVIDER.pop()]

        plan_res = self.execute_plan(args, result_ids)

        return plan_res

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

        if self.find_location(args) == self.owner:
            if self.forward is not None:
                if self.include_state:
                    return self.forward(*args, self.state)
                else:
                    return self.forward(*args)
            else:
                return self._execute_readable_plan(*args)
        else:
            if not self.is_built:
                if self.include_state:
                    self.forward(*args, self.state)
                else:
                    self.forward(*args)
            return self._execute_readable_plan(*args)

    def _update_args(
        self,
        args: List[Union[FrameworkTensorType, AbstractTensor]],
        result_ids: List[Union[str, int]],
    ):
        """Replace args and result_ids with the ones given.
        Updates the arguments ids and result ids used to execute
        the plan.
        Args:
            args: List of tensors.
            result_ids: Ids where the plan output will be stored.
        """

        arg_ids = [arg.id for arg in args]
        self.replace_ids(self.arg_ids, arg_ids)
        self.arg_ids = arg_ids

        self.replace_ids(self.result_ids, result_ids)
        self.result_ids = result_ids

    def _execute_plan(self):
        for message in self.readable_plan:
            bin_message = sy.serde.serialize(message, simplified=True)
            _ = self.owner.recv_msg(bin_message)

    def _get_plan_output(self, result_ids, return_ptr=False):
        responses = []
        for return_id in result_ids:
            response = PointerTensor(
                location=self.owner, id_at_location=return_id, owner=self, id=sy.ID_PROVIDER.pop()
            )
            responses.append(response if return_ptr else response.get())

        if len(responses) == 1:
            return responses[0]

        return responses

    def execute_plan(self, args: List, result_ids: List[Union[str, int]]):
        """Controls local or remote plan execution.

        If the plan doesn't have the plan built, first build it using the blueprint.
        Then if it has a remote location, send the plan to the remote location only the
        first time, request a remote plan execution with specific pointers and ids for
        storing the result, and return a pointer to the result of the execution.
        If the plan is local: update the plan with the result_ids and args ids given,
        run the plan and return the None message serialized.

        Args:
            args: Arguments used to run plan.
            result_ids: List of ids where the results will be stored.
        """
        # We build the plan only if needed
        if not self.is_built:
            self._build(args)

        if len(self.locations):
            plan_name = f"plan{self.id}"
            # args, _, _ = hook_args.unwrap_args_from_function(
            #     plan_name, args, {}
            # )

            worker = self.find_location(args)
            if worker.id not in self.ptr_plans.keys():
                self.ptr_plans[worker.id] = self._send(worker)
            response = self.request_execute_plan(worker, result_ids, *args)

            response = hook_args.hook_response(plan_name, response, wrap_type=FrameworkTensor[0])
            return response

        # if the plan is not to be sent then it has been requested to be executed,
        # so we update the plan with the
        # correct input and output ids and we run it
        elif not len(self.locations):
            self._update_args(args, result_ids)
            self._execute_plan()
            responses = self._get_plan_output(result_ids)
            return responses

        return sy.serde.serialize(None)

    def request_execute_plan(
        self,
        location: "sy.workers.BaseWorker",
        response_ids: List[Union[str, int]],
        *args,
        **kwargs,
    ) -> object:
        """Requests plan execution.

        Send a request to execute the plan on the remote location.

        Args:
            location: to which worker the request should be sent
            response_ids: Where the plan result should be stored remotely.
            args: Arguments used as input data for the plan.
            kwargs: Named arguments used as input data for the plan.

        Returns:
            Execution response.

        """
        args = [args, response_ids]
        command = ("execute_plan", self.ptr_plans[location.id], args, kwargs)

        response = self.owner.send_command(
            message=command, recipient=location, return_ids=response_ids
        )
        return response

    def send(self, *locations, force=False):
        """Send plan to locations.

        If the plan was not built locally it will raise an exception.
        If `force` = true plan is going to be sent either way.

        Args:
            locations: List of workers.
            force: A boolean indicating if this operation should be forced.

        Args:
            location: Workers where plan should be sent to.
        """
        if not self.is_built and not force:
            raise RuntimeError("A plan needs to be built before being sent to a worker.")

        self.locations += [self.owner.get_worker(location).id for location in locations]

        # rm duplicates
        self.locations = list(set(self.locations) - set([sy.framework.hook.local_worker.id]))

        for location in locations:
            self.ptr_plans[location.id] = self._send(location)

        return self

    def _send(self, location: "sy.workers.BaseWorker"):
        """Real send function that sends the Plan instance with its plan to location.

        Only called when the plan is built and that an execution is called, namely when it is
        necessary to send it.

        Args:
            location: Worker where plan should be sent to.
        """
        readable_plan_original = copy.deepcopy(self.readable_plan)
        for worker_id in [self.owner.id] + self.locations:
            self.replace_worker_ids(worker_id, location.id)

        state_original = self.state.copy()
        state_ids_original = self.state_ids.copy()

        self.state.send(location, garbage_collect_data=False)
        state_ptr_ids = self.state.get_id_at_location()
        self.replace_ids(self.state_ids, state_ptr_ids)
        self.state_ids = state_ptr_ids

        _ = self.owner.send(self, workers=location)

        # Deep copy the plan without using deep copy
        pointer = sy.Plan.detail(self.owner, sy.Plan.simplify(self))

        self.readable_plan = readable_plan_original
        self.state.set_(state_original)
        self.state_ids = state_ids_original
        return pointer

    def get(self) -> "Plan":
        """Mock get function.

        No call to remote worker is done, we just erase the
        information linking this plan to that remote worker.

        Returns:
            Plan.
        """

        if len(self.locations):
            self.locations = []
            self.ptr_plans = {}
        else:
            self.state.get()

        return self

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

    def create_pointer(self, owner, garbage_collect_data):
        return PointerPlan(
            location=self.owner,
            id_at_location=self.id,
            owner=owner,
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

        if len(self.locations):
            for location in self.locations:
                out += " location:" + str(location)

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
            tuple(
                plan.readable_plan
            ),  # We're not simplifying because readable_plan is already simplified
            sy.serde._simplify(plan.id),
            sy.serde._simplify(plan.arg_ids),
            sy.serde._simplify(plan.result_ids),
            sy.serde._simplify(plan.state_ids),
            sy.serde._simplify(plan.name),
            sy.serde._simplify(plan.tags),
            sy.serde._simplify(plan.description),
            plan.is_built,
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

        readable_plan, id, arg_ids, result_ids, state_ids, name, tags, description, is_built = (
            plan_tuple
        )
        id = sy.serde._detail(worker, id)
        arg_ids = sy.serde._detail(worker, arg_ids)
        result_ids = sy.serde._detail(worker, result_ids)
        state_ids = sy.serde._detail(worker, state_ids)

        plan = sy.Plan(
            owner=worker,
            id=id,
            arg_ids=arg_ids,
            result_ids=result_ids,
            readable_plan=readable_plan,  # We're not detailing, see simplify() for details
            is_built=is_built,
        )

        plan.state_ids = state_ids

        plan.name = sy.serde._detail(worker, name)
        plan.tags = sy.serde._detail(worker, tags)
        plan.description = sy.serde._detail(worker, description)

        return plan
