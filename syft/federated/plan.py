import copy
import torch

from syft.frameworks.torch.tensors.interpreters.abstract import AbstractTensor
from syft.generic import ObjectStorage
from syft.codes import MSGTYPE
import syft as sy


from typing import List
from typing import Union


def make_plan(plan_blueprint):
    """Creates a plan from a function.

    For folks who would prefer to not use a decorator, they can use this function
    to create a plan.
    """
    return func2plan(plan_blueprint)


def func2plan(plan_blueprint):
    """Converts a function to a plan.

    Converts a function containing sequential pytorch code into
    a plan object which can be sent to any arbitrary worker.
    """
    plan = Plan(owner=sy.local_worker, id=sy.ID_PROVIDER.pop(), name=plan_blueprint.__name__)
    plan.blueprint = plan_blueprint
    return plan


def method2plan(plan_blueprint):
    """Converts a method to a plan.

    Converts a method containing sequential pytorch code into
    a plan object which can be sent to any arbitrary worker.
    """
    plan = Plan(owner=sy.local_worker, id=sy.ID_PROVIDER.pop(), name=plan_blueprint.__name__)
    plan.blueprint = plan_blueprint

    @property
    def method(self: object) -> Plan:
        """
        This property is a way to catch the self of the method and give it to the plan,
        it will be provided in the future calls as this is not automatic (the structure
        of @func2plan would not keep the self during the call)

        Args:
            self (object): an instance of a class

        Returns:
            the plan which is also a callable.

        Example:
            When you have your plan and that you do
            > plan(*args)
            First the property is call with the part "plan" and self is caught, plan is
            returned
            Then plan is called with "(*args)" and in the __call__ function of plan the
            self parameter is re-inserted
        """
        plan.self = self

        return plan

    return method


class Plan(ObjectStorage):
    """A Plan store a sequence of torch operations, just like a function.

    A Plan is intended to store a sequence of torch operations, just like a function,
    but it allows to send this sequence of operations to remote workers and to keep a
    reference to it. This way, to compute remotely this sequence of operations on some remote
    input referenced through pointers, instead of sending multiple messages you need now to send a
    single message with the references of the plan and the pointers.
    """

    def __init__(
        self,
        id: Union[str, int],
        owner: "sy.workers.BaseWorker",
        name: str = "",
        arg_ids: List[Union[str, int]] = None,
        result_ids: List[Union[str, int]] = None,
        blueprint: callable = None,
        readable_plan: List = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Plan instance info
        self.id = id
        self.name = name
        self.owner = owner

        # Info about the plan stored
        self.plan = list()
        self.readable_plan = readable_plan if readable_plan is not None else []
        self.arg_ids = arg_ids if arg_ids is not None else []
        self.result_ids = result_ids if result_ids is not None else []
        self.owner_when_built = None

        # Pointing info towards a remote plan
        self.locations = []
        self.ptr_plans = {}

        self.tags = None
        self.description = None
        self.blueprint = blueprint

        # For methods
        self.self = None

    @property
    def _known_workers(self):
        return self.owner._known_workers

    def send_msg(self, *args, **kwargs):
        return self.owner.send_msg(*args, **kwargs)

    def request_obj(self, *args, **kwargs):
        return self.owner.request_obj(*args, **kwargs)

    def _recv_msg(self, bin_message: bin):
        """Upon reception, a Plan stores all commands which can be executed lazily.
        Args:
            bin_message: the message of a command received.
        Returns:
            The None message serialized to specify the command was received.
        """
        (some_type, (msg_type, contents)) = sy.serde.deserialize(bin_message, detail=False)

        if msg_type != MSGTYPE.OBJ:
            self.plan.append(bin_message)
            self.readable_plan.append((some_type, (msg_type, contents)))

        # we can't receive the results of a plan without
        # executing it. So, execute the plan.
        if msg_type in (MSGTYPE.OBJ_REQ, MSGTYPE.IS_NONE, MSGTYPE.GET_SHAPE):
            return self.__call__()

        return sy.serde.serialize(None)

    def build_plan(self, args: List):
        """Builds a plan.

        The plan must be built with some input data, here `args`. When they
        are provided, they are sent to the plan worker, which executes its
        blueprint: each command of the blueprint is catched by _recv_msg
        and is used to fill the plan.

        Args:
            param: Input data.
        """
        # The ids of args of the first call, which should be updated when
        # the function is called with new args
        self.arg_ids = list()
        local_args = list()
        for arg in args:
            # Send only tensors (in particular don't send the "self" for methods)
            # in the case of a method.
            if isinstance(arg, torch.Tensor):
                self.owner.register_obj(arg)
                arg = arg.send(self)
                arg.child.garbage_collect_data = False
                self.arg_ids.append(arg.id_at_location)
            local_args.append(arg)

        res_ptr = self.blueprint(*local_args)
        res_ptr.child.garbage_collect_data = False

        worker = self.find_location(args)

        self.replace_worker_ids(worker.id, self.owner.id)

        # The id where the result should be stored
        self.result_ids = [res_ptr.id_at_location]

        # Store owner that built the plan
        self.owner_when_built = self.owner

    def find_location(self, args):
        """
        Return location if args contain pointers else the local worker
        """
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if hasattr(arg, "child") and isinstance(arg.child, sy.PointerTensor):
                    return arg.location
        return sy.hook.local_worker

    def copy(self):
        """Creates a copy of a plan."""
        plan = Plan(
            sy.ID_PROVIDER.pop(),
            self.owner,
            self.name,
            arg_ids=self.arg_ids,
            result_ids=self.result_ids,
            readable_plan=self.readable_plan,
            blueprint=self.blueprint,
        )
        plan.replace_ids(
            from_ids=plan.arg_ids, to_ids=plan.arg_ids, from_worker=self.id, to_worker=plan.id
        )
        plan.replace_ids(
            from_ids=plan.result_ids, to_ids=plan.result_ids, from_worker=self.id, to_worker=plan.id
        )
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

        for i, item in enumerate(obj):
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

        return _obj

    def __call__(self, *args, **kwargs):
        """Calls a plan.

        Calls a plan execution with some arguments, and specify the ids where the result
        should be stored.

        Returns:
            The pointer to the result of the execution if the plan was already sent,
            else the None message serialized.
        """
        assert len(kwargs) == 0, "kwargs not supported for plan"
        result_ids = [sy.ID_PROVIDER.pop()]
        # Support for method hooked in plans
        if self.self is not None:
            args = [self.self] + list(args)
        return self.execute_plan(args, result_ids)

    def _update_args(
        self, args: List[Union[torch.Tensor, AbstractTensor]], result_ids: List[Union[str, int]]
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
            response = sy.PointerTensor(
                location=self.owner, id_at_location=return_id, owner=self, id=sy.ID_PROVIDER.pop()
            )
            responses.append(response if return_ptr else response.get())

        if len(responses) == 1:
            return responses[0]

        return responses

    def _execute_plan_locally(self, result_ids, *args, **kwargs):
        tensor_args = self._keep_only_tensor_args(args)
        self._update_args(tensor_args, result_ids)
        self._execute_plan()
        responses = self._get_plan_output(result_ids)
        return responses

    def _keep_only_tensor_args(self, args):
        """Keeps only Tensors in args. This step is needed for method plans."""
        tensor_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_args.append(arg)
        return tensor_args

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
        first_run = self.readable_plan == []
        if first_run:
            self.build_plan(args)

        if len(self.locations) > 0:
            worker = self.find_location(args)
            if worker.id not in self.ptr_plans.keys():
                self.ptr_plans[worker.id] = self._send(worker)
            response = self.request_execute_plan(worker, result_ids, *args)

            return response

        # If the plan is local, we execute the plan and return the response
        if len(self.locations) == 0 and self.owner == sy.hook.local_worker:
            return self._execute_plan_locally(result_ids, *args)

        # if the plan is not to be sent but is not local (ie owned by the local worker)
        # then it has been requested to be executed, so we update the plan with the
        # correct input and output ids and we run it
        elif len(self.locations) == 0 and self.owner != sy.hook.local_worker:
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
        args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        args = [args, response_ids]
        command = ("execute_plan", self.ptr_plans[location.id], args, kwargs)

        response = self.owner.send_command(
            message=command, recipient=location, return_ids=response_ids
        )
        return response

    def send(self, *locations):
        """Mock send function that only specify that the Plan will have to be sent to location.

        When one calls .send(), this doesn't trigger a call to remote workers, but
        just stores "a promise" that it will be sent (with _send()) later when the plan in
        called (and built)

        Args:
            location: Workers where plan should be sent to.
        """
        self.locations += [self.owner.get_worker(location).id for location in locations]
        # rm duplicates
        self.locations = list(set(self.locations))
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
        _ = self.owner.send(self, workers=location)

        # Deep copy the plan without using deep copy
        pointer = sy.serde._detail_plan(self.owner, sy.serde._simplify_plan(self))

        # readable_plan, id, arg_ids, result_ids, name, tags, description = plan_tuple
        self.readable_plan = readable_plan_original
        return pointer

    def get(self) -> "Plan":
        """Mock get function.

        No call to remote worker is done, we just erase the
        information linking this plan to that remote worker.

        Returns:
            Plan.
        """
        # self.replace_worker_ids(self.location.id, self.owner.id)

        self.locations = []
        self.ptr_plans = {}

        return self

    def describe(self, description: str) -> "Plan":
        self.description = description
        return self

    def tag(self, *_tags: List) -> "Plan":
        if self.tags is None:
            self.tags = set()

        for new_tag in _tags:
            self.tags.add(new_tag)
        return self

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

        if len(self.readable_plan) > 0:
            out += " built"

        out += ">"

        return out
