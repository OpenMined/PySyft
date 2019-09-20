import syft as sy

from syft.exceptions import WorkerNotFoundException
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.object import AbstractObject
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.generic.pointers.pointer_protocol import PointerProtocol
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker

from typing import List, Union


class Protocol(AbstractObject):
    """
    A Protocol coordinates a sequence of Plans, deploys them on distant workers
    and run them in a single pass.

    It's a high level object which contains the logic of a complex computation
    distributed across several workers. The main feature of Protocol is the
    ability to be sent / searched / fetched back between workers, and finally
    deployed to identified workers. So a user can design a protocol, upload it
    to a cloud worker, and any other workers will be able to search for it,
    download it, and apply the computation program it contains on the workers
    that it is connected to.


    Args:
        plans: a list of pairs (worker, plan). "worker" can be either a real
            worker or a worker id or a string to represent a fictive worker. This
            last case can be used at creation to specify that two plans should be
            owned (or not owned) by the same worker at deployment. "plan" can
            either be a Plan or a PointerPlan.
        id: the Protocol id
        owner: the Protocol owner
        tags: the Protocol tags (used for search)
        description: the Protocol description
    """

    def __init__(
        self,
        plans: List = None,
        id: int = None,
        owner: "sy.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
    ):
        owner = owner or sy.framework.hook.local_worker
        super(Protocol, self).__init__(id, owner, tags, description, child=None)

        self.plans = plans or list()
        self.workers_resolved = len(self.plans) and all(
            isinstance(w, AbstractWorker) for w, p in self.plans
        )
        self.location = None

    def deploy(self, *workers):
        """
        Calling .deploy() sends the plans to the designated workers.

        This is done in 2 phases: first, we map the fictive workers provided at creation
        (named by strings) to the provided workers, and second, we send the corresponding
        plans to each of them.
        For the first phase, either there is exactly one real worker per plan or one real
        worker per fictive_worker. _resolve_workers replaces the fictive workers by the
        real ones.
        """
        if self.workers_resolved:
            raise RuntimeError(
                f"This protocol is already deployed to {', '.join(worker.id for worker, plan in self.plans)}."
            )

        n_workers = len(set(worker for worker, plan in self.plans))

        if len(self.plans) != len(workers) != n_workers:
            raise RuntimeError(
                f"This protocol is designed for {n_workers} workers, but {len(workers)} were provided."
            )

        self._resolve_workers(workers)

        for worker, plan in self.plans:
            plan.send(worker)

        return self

    def run(self, *args, **kwargs):
        """
        Run the protocol by executing the plans sequentially

        The input args provided are sent to the first plan location. This first plan is
        run and its output is moved to the second plan location, and so on. The final
        result is returned after all plans have run, and it is composed of pointers to
        the last plan location.
        """
        self._assert_is_resolved()

        # This is an alternative to having PointerProtocol, when we send the protocol.
        if self.location is not None:
            location = Protocol.find_args_location(args)
            if location != self.location:
                raise RuntimeError(
                    f"This protocol has been sent to {self.location.id}, but you provided "
                    f"local arguments or pointers to {location.id}."
                )

            print("send remote run request to", self.location.id)
            response = self.request_remote_run(location, args, kwargs)
            return response

        # Local and sequential coordination of the plan execution
        previous_worker_id = None
        response = None
        for worker, plan in self.plans:
            # Transmit the args to the next worker if it's a different one % the previous
            if None is not previous_worker_id != worker.id:
                print("move", previous_worker_id, " -> ", worker.id)
                args = [arg.move(worker) for arg in args]
            else:
                print("send", worker.id)
                args = [arg.send(worker) for arg in args]

            previous_worker_id = worker.id

            response = plan(*args)

            args = response if isinstance(response, tuple) else (response,)

        return response

    def request_remote_run(self, location: "sy.workers.BaseWorker", args, kwargs) -> object:
        """Requests protocol execution.

        Send a request to execute the protocol on the remote location.

        Args:
            location: to which worker the request should be sent
            args: Arguments used as input data for the protocol.
            kwargs: Named arguments used as input data for the protocol.

        Returns:
            Execution response.

        """
        plan_name = f"plan{self.id}"
        args, _, _ = hook_args.unwrap_args_from_function(plan_name, args, {})

        # return_ids = kwargs.get("return_ids", {})
        command = ("run", self.id, args, kwargs)

        response = self.owner.send_command(
            message=command, recipient=location  # , return_ids=return_ids
        )
        response = hook_args.hook_response(plan_name, response, wrap_type=FrameworkTensor[0])
        return response

    @staticmethod
    def find_args_location(args):
        """
        Return location if args contain pointers else the local worker
        """
        for arg in args:
            if isinstance(arg, FrameworkTensor):
                if hasattr(arg, "child") and isinstance(arg.child, PointerTensor):
                    return arg.location
        return sy.framework.hook.local_worker

    def send(self, location):
        """
        Send a protocol to a worker, to be fetched by other workers
        """

        # If the workers have not been assigned, then the plans are still local
        # and should be sent together with the protocol to the location
        if not self.workers_resolved:
            for _, plan in self.plans:
                plan.send(location)
        # Else, the plans are already deployed and we don't move them

        self.owner.send_obj(obj=self, location=location)

        self.location = location

    @staticmethod
    def simplify(protocol: "Protocol") -> tuple:
        """
        This function takes the attributes of a Protocol and saves them in a tuple
        Args:
            protocol (Protocol): a Protocol object
        Returns:
            tuple: a tuple holding the unique attributes of the Protocol object
        """
        plans_reference = []
        for worker, plan in protocol.plans:
            if isinstance(plan, sy.Plan):
                plan_id = plan.id
            elif isinstance(plan, sy.PointerPlan):
                plan_id = plan.id_at_location
            else:
                raise TypeError("This is not a valid Plan")

            if isinstance(worker, str):
                worker_id = worker
            else:
                worker_id = worker.id

            plans_reference.append((worker_id, plan_id))

        return (
            sy.serde._simplify(protocol.id),
            sy.serde._simplify(protocol.tags),
            sy.serde._simplify(protocol.description),
            sy.serde._simplify(plans_reference),
            sy.serde._simplify(protocol.workers_resolved),
        )

    @staticmethod
    def detail(worker: BaseWorker, protocol_tuple: tuple) -> "Protocol":
        """This function reconstructs a Protocol object given its attributes in the form of a tuple.
        Args:
            worker: the worker doing the deserialization
            protocol_tuple: a tuple holding the attributes of the Protocol
        Returns:
            protocol: a Protocol object
        """

        id, tags, description, plans_reference, workers_resolved = map(
            lambda o: sy.serde._detail(worker, o), protocol_tuple
        )

        plans = []
        for owner_id, plan_id in plans_reference:
            if workers_resolved:
                plan_owner = worker.get_worker(owner_id, fail_hard=True)
                plan_pointer = worker.request_search(plan_id, location=plan_owner)[0]
                worker.register_obj(plan_pointer)
                plans.append((plan_owner, plan_pointer))
            else:
                try:
                    plan_owner = worker.get_worker(owner_id, fail_hard=True)
                    plan_pointer = worker.request_search(plan_id, location=plan_owner)[0]
                    plan = plan_pointer.get()
                except WorkerNotFoundException:
                    plan = worker.get_obj(plan_id)
                plans.append((worker.id, plan))

        protocol = sy.Protocol(plans=plans, id=id, owner=worker, tags=tags, description=description)

        return protocol

    def create_pointer(self, owner, garbage_collect_data):
        return PointerProtocol(
            location=self.owner,
            id_at_location=self.id,
            owner=owner,
            garbage_collect_data=garbage_collect_data,
        )

    def __repr__(self):
        repr = f"<Protocol id:{self.id} owner:{self.owner.id}{' resolved' if self.workers_resolved else ''}>"
        for worker, plan in self.plans:
            repr += "\n - "
            if isinstance(worker, str):
                repr += worker
            else:
                repr += str(worker.id)
            repr += ": "
            repr += plan.__repr__()
        return repr

    def __str__(self) -> str:
        return self.__repr__()

    def _assert_is_resolved(self):
        if not self.workers_resolved:
            raise RuntimeError(
                "Plans have not been allocated to existing workers. Call deploy(*workers) to do so."
            )

    def _resolve_workers(self, workers):
        """Map the abstract workers (named by strings) to the provided workers and
        update the plans accordingly"""
        # If there is an exact one-to-one mapping, just iterate and keep the order
        # provided when assigning the workers
        if len(workers) == len(self.plans):
            self.plans = [(worker, plan) for (_, plan), worker in zip(self.plans, workers)]

        # Else, there are duplicates in the self.plans keys and we need to build
        # a small map
        # Example:
        #   protocol.plans == [("w1", plan1), ("w2", plan2), ("w1", plan3)
        #   protocol.deploy(alice, bob)
        else:
            worker_map = {
                abstract_worker_name: worker
                for worker, abstract_worker_name in zip(
                    workers, set(name for name, plan in self.plans)
                )
            }
            self.plans = [(worker_map[name], plan) for name, plan in self.plans]

        self.workers_resolved = True
