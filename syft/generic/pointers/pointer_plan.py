from typing import List
from typing import Union

import syft as sy
from syft.generic.frameworks.hook import hook_args
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.frameworks.types import FrameworkTensor
from syft.workers.abstract import AbstractWorker


class PointerPlan(ObjectPointer):
    """
    The PointerPlan keeps a reference to a remote Plan.

    It allows to:
    - __call__ an evaluation of the remote plan
    - get the remote plan

    It's a simplification compared to the current hybrid state of
    Plans which can be seen as pointers, which is ambiguous.
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

    def __call__(self, *args, **kwargs):
        """
        Transform the call on the pointer in a request to evaluate the
        remote plan
        """
        result_ids = [sy.ID_PROVIDER.pop()]

        response = self.request_execute_plan(self.location, result_ids, *args)

        return response

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
            response_ids: where the result should be stored
            args: arguments used as input data for the plan
            kwargs: named arguments used as input data for the plan

        Returns:
            Execution response
        """
        plan_name = f"plan{self.id}"
        # args, _, _ = hook_args.unwrap_args_from_function(
        #     plan_name, args, {}
        # )
        args = [args, response_ids]

        command = ("execute_plan", self.id_at_location, args, kwargs)

        response = self.owner.send_command(
            message=command, recipient=location, return_ids=response_ids
        )
        response = hook_args.hook_response(plan_name, response, wrap_type=FrameworkTensor[0])
        response.garbage_collect_data = False
        return response

    def get(self, deregister_ptr: bool = True):
        """
        This is an alias to fetch_plan, to behave like a pointer
        """
        copy = not deregister_ptr
        plan = self.owner.fetch_plan(self.id_at_location, self.location, copy=copy)
        return plan

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

            ptr = PointerPlan(
                location=location,
                id_at_location=id_at_location,
                owner=worker,
                garbage_collect_data=garbage_collect_data,
                id=obj_id,
            )

            return ptr

    def wrap(self):
        return self
