from typing import List
from typing import Union

import syft as sy
from syft.generic.frameworks.hook import hook_args
from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.frameworks.types import FrameworkTensor
from syft.messaging.message import ForceObjectDeleteMessage
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

        self._locations = []
        self._ids_at_location = []

        super().__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            garbage_collect_data=garbage_collect_data,
            id=id,
            tags=tags,
            description=description,
        )

    # Make PointerPlan compatible with multi pointers
    @property
    def location(self):
        n_locations = len(self._locations)
        if n_locations != 1:
            return self._locations
        else:
            return self._locations[0]

    @location.setter
    def location(self, new_location: Union[AbstractWorker, List[AbstractWorker]]):
        if isinstance(new_location, (list, tuple)):
            self._locations = new_location
        else:
            self._locations = [new_location]

    @property
    def id_at_location(self):
        n_ids = len(self._ids_at_location)
        if n_ids != 1:
            return self._ids_at_location
        else:
            return self._ids_at_location[0]

    @id_at_location.setter
    def id_at_location(self, new_id_at_location):
        if isinstance(new_id_at_location, (list, tuple)):
            self._ids_at_location = new_id_at_location
        else:
            self._ids_at_location = [new_id_at_location]

    def __call__(self, *args, **kwargs):
        """
        Transform the call on the pointer in a request to evaluate the
        remote plan
        """
        if len(self._locations) > 1 and isinstance(args[0], sy.MultiPointerTensor):
            responses = {}
            for location in self._locations:
                child_args = [
                    x.child[location.id] if isinstance(x, sy.MultiPointerTensor) else x
                    for x in args
                ]
                responses[location.id] = self.__call__(*child_args, **kwargs)

            return responses

        if len(self._locations) == 1:
            location = self.location
        else:
            location = args[0].location

        result_ids = [sy.ID_PROVIDER.pop()]
        response = self.request_run_plan(location, result_ids, *args)

        return response

    def parameters(self) -> List:
        """Return a list of pointers to the plan parameters"""

        assert (
            len(self._locations) == 1
        ), ".parameters() for PointerPlan with > 1 locations is currently not implemented."
        # TODO implement this feature using MultiPointerTensor

        location = self._locations[0]
        id_at_location = self._ids_at_location[0]

        pointers = self.owner.send_command(
            cmd_name="parameters", target=id_at_location, recipient=location
        )

        for pointer in pointers:
            pointer.garbage_collect_data = False

        return [pointer.wrap() for pointer in pointers]

    def request_run_plan(
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

        args = [args, response_ids]

        if location not in self._locations:
            raise RuntimeError(
                f"Requested to run a plan on {location.id} but pointer location(s) is/are",
                self._locations,
            )

        # look for the relevant id in the list of ids
        id_at_location = None
        for loc, id_at_loc in zip(self._locations, self._ids_at_location):
            if loc == location:
                id_at_location = id_at_loc
                break

        response = self.owner.send_command(
            cmd_name="run",
            target=id_at_location,
            args_=tuple(args),
            recipient=location,
            return_ids=tuple(response_ids),
        )
        response = hook_args.hook_response(plan_name, response, wrap_type=FrameworkTensor[0])
        if isinstance(response, (list, tuple)):
            for r in response:
                r.garbage_collect_data = False
        else:
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
    def simplify(worker: AbstractWorker, ptr: "PointerPlan") -> tuple:

        return (
            sy.serde.msgpack.serde._simplify(worker, ptr.id),
            sy.serde.msgpack.serde._simplify(worker, ptr.id_at_location),
            sy.serde.msgpack.serde._simplify(worker, ptr.location.id),
            sy.serde.msgpack.serde._simplify(worker, ptr.tags),
            ptr.garbage_collect_data,
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PointerPlan":
        # TODO: fix comment for this and simplifier
        obj_id, id_at_location, worker_id, tags, garbage_collect_data = tensor_tuple

        obj_id = sy.serde.msgpack.serde._detail(worker, obj_id)
        id_at_location = sy.serde.msgpack.serde._detail(worker, id_at_location)
        worker_id = sy.serde.msgpack.serde._detail(worker, worker_id)
        tags = sy.serde.msgpack.serde._detail(worker, tags)

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
                tags=tags,
                garbage_collect_data=garbage_collect_data,
                id=obj_id,
            )

            return ptr

    def wrap(self):
        return self

    def __str__(self):
        """Returns a string version of this pointer.

        Example:
            For single pointers:
            > [PointerPlan | me:33873097403 -> dan:72165846784]

            Or for multi pointers:
            > [PointerPlan | me:55894304374
                 -> alice:72165846784
                 -> bob:72165846784
            ]
        """
        type_name = type(self).__name__
        out = f"[" f"{type_name} | " f"{str(self.owner.id)}:{self.id}"
        if len(self._locations) == 1:
            out += f" -> {str(self.location.id)}:{self.id_at_location}"
        else:
            for location, id_at_location in zip(self.location, self.id_at_location):
                out += f"\n\t -> {str(location.id)}:{id_at_location}"
            out += "\n"
        out += "]"

        if self.tags is not None and len(self.tags):
            out += "\n\tTags: "
            for tag in self.tags:
                out += str(tag) + " "

        if self.description is not None:
            out += "\n\tDescription: " + str(self.description).split("\n")[0] + "..."

        return out

    def __del__(self):
        """This method garbage collects the object this pointer is pointing to.
        By default, PySyft assumes that every object only has one pointer to it.
        Thus, if the pointer gets garbage collected, we want to automatically
        garbage collect the object being pointed to.
        """
        if self.garbage_collect_data:
            for id_at_location, location in zip(self._ids_at_location, self._locations):
                self.owner.send_msg(ForceObjectDeleteMessage(id_at_location), location)
