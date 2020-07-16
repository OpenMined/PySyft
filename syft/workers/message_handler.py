from typing import List
from typing import Union

import syft as sy
from syft import codes

from syft.execution.computation import ComputationAction
from syft.execution.communication import CommunicationAction
from syft.generic.abstract.hookable import map_chain_call
from syft.generic.abstract.message_handler import AbstractMessageHandler
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.messaging.message import TensorCommandMessage
from syft.messaging.message import WorkerCommandMessage
from syft.messaging.message import ForceObjectDeleteMessage
from syft.messaging.message import GetShapeMessage
from syft.messaging.message import IsNoneMessage
from syft.messaging.message import ObjectMessage
from syft.messaging.message import ObjectRequestMessage
from syft.messaging.message import PlanCommandMessage
from syft.messaging.message import SearchMessage

from syft.exceptions import GetNotPermittedError
from syft.exceptions import ObjectNotFoundError
from syft.exceptions import PlanCommandUnknownError
from syft.exceptions import ResponseSignatureError


class BaseMessageHandler(AbstractMessageHandler):
    def __init__(self, object_store, worker):
        super().__init__(object_store)
        self.worker = worker

        self.plan_routing_table = {
            codes.PLAN_CMDS.FETCH_PLAN: self._fetch_plan_remote,
            codes.PLAN_CMDS.FETCH_PROTOCOL: self._fetch_protocol_remote,
        }

    def init_routing_table(self):
        return {
            TensorCommandMessage: self.execute_tensor_command,
            PlanCommandMessage: self.execute_plan_command,
            WorkerCommandMessage: self.execute_worker_command,
            ObjectMessage: self.handle_object_msg,
            ObjectRequestMessage: self.respond_to_obj_req,
            ForceObjectDeleteMessage: self.handle_force_delete_object_msg,
            IsNoneMessage: self.is_object_none,
            GetShapeMessage: self.handle_get_shape_message,
            SearchMessage: self.respond_to_search,
        }

    def execute_tensor_command(self, cmd: TensorCommandMessage) -> PointerTensor:
        if isinstance(cmd.action, ComputationAction):
            return self.execute_computation_action(cmd.action)
        else:
            return self.execute_communication_action(cmd.action)

    def execute_computation_action(self, action: ComputationAction) -> PointerTensor:
        """
        Executes commands received from other workers.
        Args:
            message: A tuple specifying the command and the args.
        Returns:
            The result or None if return_value is False.
        """

        op_name = action.name
        _self = action.target
        args_ = action.args
        kwargs_ = action.kwargs
        return_ids = action.return_ids
        return_value = action.return_value

        # Handle methods
        if _self is not None:
            if type(_self) == int:
                _self = self.get_obj(_self)
                if _self is None:
                    return
            elif isinstance(_self, str):
                if _self == "self":
                    _self = self.worker
                else:
                    res: list = self.worker.search(_self)
                    assert (
                        len(res) == 1
                    ), f"Searching for {_self} on {self.worker.id}. /!\\ {len(res)} found"
                    _self = res[0]
            if sy.framework.is_inplace_method(op_name):
                # TODO[jvmancuso]: figure out a good way to generalize the
                # above check (#2530)
                getattr(_self, op_name)(*args_, **kwargs_)
                return
            else:
                try:
                    response = getattr(_self, op_name)(*args_, **kwargs_)
                except TypeError:
                    # TODO Andrew thinks this is gross, please fix. Instead need to
                    # properly deserialize strings
                    new_args = [
                        arg.decode("utf-8") if isinstance(arg, bytes) else arg for arg in args_
                    ]
                    response = getattr(_self, op_name)(*new_args, **kwargs_)
        # Handle functions
        else:
            # At this point, the command is ALWAYS a path to a
            # function (i.e., torch.nn.functional.relu). Thus,
            # we need to fetch this function and run it.

            sy.framework.command_guard(op_name)

            paths = op_name.split(".")
            command = self.worker
            for path in paths:
                command = getattr(command, path)

            response = command(*args_, **kwargs_)

        # some functions don't return anything (such as .backward())
        # so we need to check for that here.
        if response is not None:
            # Register response and create pointers for tensor elements
            try:
                response = hook_args.register_response(
                    op_name, response, list(return_ids), self.worker
                )
                # TODO: Does this mean I can set return_value to False and still
                # get a response? That seems surprising.
                if return_value or isinstance(response, (int, float, bool, str)):
                    return response
                else:
                    return None
            except ResponseSignatureError:
                return_id_provider = sy.ID_PROVIDER
                return_id_provider.set_next_ids(return_ids, check_ids=False)
                return_id_provider.start_recording_ids()
                response = hook_args.register_response(
                    op_name, response, return_id_provider, self.worker
                )
                new_ids = return_id_provider.get_recorded_ids()
                raise ResponseSignatureError(new_ids)

    def execute_communication_action(self, action: CommunicationAction) -> PointerTensor:
        owner = action.target.owner
        destinations = [self.worker.get_worker(id_) for id_ in action.args]
        kwargs_ = action.kwargs

        if owner != self.worker:
            return None
        else:
            obj = self.get_obj(action.target.id)
            response = owner.send(obj, *destinations, **kwargs_)
            response.garbage_collect_data = False
            if kwargs_.get("requires_grad", False):
                response = hook_args.register_response(
                    "send", response, [sy.ID_PROVIDER.pop()], self.worker
                )
            else:
                self.object_store.rm_obj(action.target.id)
            return response

    def handle_object_msg(self, obj_msg: ObjectMessage):
        # This should be a good seam for separating Workers from ObjectStore (someday),
        # so that Workers have ObjectStores instead of being ObjectStores. That would open
        # up the possibility of having a separate ObjectStore for each user, or for each
        # Plan/Protocol, etc. As Syft moves toward multi-tenancy with Grid and so forth,
        # that will probably be useful for providing security and permissioning. In that
        # future, this might look like `self.object_store.set_obj(obj_msg.object)`
        """Receive an object from a another worker

        Args:
            obj: a Framework Tensor or a subclass of an AbstractTensor with an id
        """
        obj = obj_msg.object

        self.object_store.set_obj(obj)

        if isinstance(obj, FrameworkTensor):
            tensor = obj
            if (
                tensor.requires_grad
                and tensor.origin is not None
                and tensor.id_at_origin is not None
            ):
                tensor.register_hook(
                    tensor.trigger_origin_backward_hook(tensor.origin, tensor.id_at_origin)
                )

    def respond_to_obj_req(self, msg: ObjectRequestMessage):
        """Returns the deregistered object from registry.

        Args:
            request_msg (tuple): Tuple containing object id, user credentials and reason.
        """
        obj_id = msg.object_id
        user = msg.user
        reason = msg.reason

        obj = self.get_obj(obj_id)

        permitted = all(map_chain_call(obj, "allow", user=user))
        if not permitted:
            raise GetNotPermittedError()
        else:
            self.object_store.de_register_obj(obj)
            return obj

    def handle_force_delete_object_msg(self, msg: ForceObjectDeleteMessage):
        for object_id in msg.object_ids:
            self.object_store.force_rm_obj(object_id)

    def is_object_none(self, msg):
        obj_id = msg.object_id
        if obj_id not in self.object_store._objects:
            # If the object is not present on the worker, raise an error
            raise ObjectNotFoundError(obj_id, self)
        obj = self.get_obj(msg.object_id)
        return obj is None

    def handle_get_shape_message(self, msg: GetShapeMessage) -> List:
        """
        Returns the shape of a tensor casted into a list, to bypass the serialization of
        a torch.Size object.

        Args:
            tensor: A torch.Tensor.

        Returns:
            A list containing the tensor shape.
        """
        tensor = self.get_obj(msg.tensor_id)
        return list(tensor.shape)

    def respond_to_search(self, msg: SearchMessage) -> List[PointerTensor]:
        """
        When remote worker calling search on this worker, forwarding the call and
        replace found elements by pointers
        """
        query = msg.query
        objects = self.worker.search(query)
        results = []
        for obj in objects:
            # set garbage_collect_data to False because if we're searching
            # for a tensor we don't own, then it's probably someone else's
            # decision to decide when to delete the tensor.
            ptr = obj.create_pointer(
                garbage_collect_data=False, owner=sy.local_worker, tags=obj.tags
            )

            # Wrap only if the pointer points to a tensor.
            # If it points to a generic object, do not wrap.
            if isinstance(ptr, PointerTensor):
                ptr = ptr.wrap()

            results.append(ptr)

        return results

    def get_obj(self, obj_id: Union[str, int]) -> object:
        """Returns the object from registry.

        Look up an object from the registry using its ID.

        Args:
            obj_id: A string or integer id of an object to look up.
        """
        obj = self.object_store.get_obj(obj_id)

        # An object called with get_obj will be "with high probability" serialized
        # and sent back, so it will be GCed but remote data is any shouldn't be
        # deleted
        if hasattr(obj, "child") and hasattr(obj.child, "set_garbage_collect_data"):
            obj.child.set_garbage_collect_data(value=False)

        if hasattr(obj, "private") and obj.private:
            return None

        return obj

    def execute_plan_command(self, msg: PlanCommandMessage):
        """Executes commands related to plans.

        This method is intended to execute all commands related to plans and
        avoiding having several new message types specific to plans.

        Args:
            msg: A PlanCommandMessage specifying the command and args.
        """
        command_name = msg.command_name
        args_ = msg.args

        try:
            command = self.plan_routing_table[command_name]
        except KeyError:
            raise PlanCommandUnknownError(command_name)

        return command(*args_)

    def _fetch_plan_remote(self, plan_id: Union[str, int], copy: bool) -> "Plan":  # noqa: F821
        """Fetches a copy of a the plan with the given `plan_id` from the worker registry.

        This method is executed for remote execution.

        Args:
            plan_id: A string indicating the plan id.

        Returns:
            A plan if a plan with the given `plan_id` exists. Returns None otherwise.
        """
        if plan_id in self.object_store._objects:
            candidate = self.object_store.get_obj(plan_id)
            if isinstance(candidate, sy.Plan):
                if copy:
                    return candidate.copy()
                else:
                    return candidate

        return None

    def _fetch_protocol_remote(
        self, protocol_id: Union[str, int], copy: bool
    ) -> "Protocol":  # noqa: F821
        """
        Target function of fetch_protocol, find and return a protocol
        """
        if protocol_id in self.object_store._objects:

            candidate = self.object_store.get_obj(protocol_id)
            if isinstance(candidate, sy.Protocol):
                return candidate

        return None

    def execute_worker_command(self, message: tuple):
        """Executes commands received from other workers.

        Args:
            message: A tuple specifying the command and the args.

        Returns:
            A pointer to the result.
        """
        command_name = message.command_name
        args_, kwargs_, return_ids = message.message

        response = getattr(self.worker, command_name)(*args_, **kwargs_)
        #  TODO [midokura-silvia]: send the tensor directly
        #  TODO this code is currently necessary for the async_fit method in websocket_client.py
        if isinstance(response, FrameworkTensor):
            self.object_store.register_obj(obj=response, obj_id=return_ids[0])
            return None
        return response
