from typing import List
from typing import Union

import syft as sy
from syft.generic.frameworks.hook import hook_args
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.pointer_plan import PointerPlan
from syft.generic.frameworks.types import FrameworkTensor
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker


class PointerProtocol(ObjectPointer):
    """
    The PointerProtocol keeps a reference to a remote Protocol.

    It allows to:
    - run an evaluation of the remote protocol
    - get the remote protocol
    """

    def __init__(
        self,
        location: "AbstractWorker" = None,
        id_at_location: Union[str, int] = None,
        owner: "AbstractWorker" = None,
        garbage_collect_data: bool = True,
        id: Union[str, int] = None,
        tags: List[str] = None,
        description: str = None,
    ):
        if owner is None:
            owner = sy.framework.hook.local_worker

        super().__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            garbage_collect_data=garbage_collect_data,
            id=id,
            tags=tags,
            description=description,
        )

    def run(self, *args, **kwargs):
        """
        Call a run on the remote protocol
        """
        location = sy.Protocol.find_args_location(args)
        if location != self.location:
            raise RuntimeError(
                f"This protocol has been sent to {self.location.id}, but you provided "
                f"local arguments or pointers to {location.id}."
            )
        response = self.request_remote_run(location, args, kwargs)
        return response

    def request_remote_run(self, location: "BaseWorker", args, kwargs) -> object:
        """Requests remote protocol execution.

        Send a request to execute the protocol on the remote location.

        Args:
            location: to which worker the request should be sent
            args: arguments used as input data for the protocol
            kwargs: named arguments used as input data for the protocol

        Returns:
            Execution response
        """

        plan_name = f"plan{self.id}"
        args, _, _ = hook_args.unwrap_args_from_function(plan_name, args, {})

        # return_ids = kwargs.get("return_ids", {})
        command = ("run", self.id_at_location, args, kwargs)

        response = self.owner.send_command(
            message=command, recipient=location  # , return_ids=return_ids
        )
        response = hook_args.hook_response(plan_name, response, wrap_type=FrameworkTensor[0])
        return response

    def get(self, deregister_ptr: bool = True):
        """
        This is an alias to fetch_protocol, to behave like a pointer
        """
        copy = not deregister_ptr
        protocol = self.owner.fetch_protocol(self.id_at_location, self.location, copy=copy)
        return protocol

    @staticmethod
    def simplify(ptr: "PointerPlan") -> tuple:

        return (ptr.id, ptr.id_at_location, ptr.location.id, ptr.garbage_collect_data)

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PointerPlan":
        # TODO: fix comment for this and simplifier
        obj_id, id_at_location, worker_id, garbage_collect_data = tensor_tuple

        if isinstance(worker_id, bytes):
            worker_id = worker_id.decode()

        # If the pointer received is pointing at the current worker, we load the tensor instead
        if worker_id == worker.id:
            plan = worker.get_obj(id_at_location)

            return plan
        # Else we keep the same Pointer
        else:
            location = sy.hook.local_worker.get_worker(worker_id)

            ptr = PointerProtocol(
                location=location,
                id_at_location=id_at_location,
                owner=worker,
                garbage_collect_data=garbage_collect_data,
                id=obj_id,
            )

            return ptr

    def wrap(self):
        return self
