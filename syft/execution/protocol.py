import warnings
from typing import List, Optional, Tuple, Union, Set

import syft as sy
from syft.exceptions import WorkerNotFoundException
from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.object import AbstractObject
from syft.generic.pointers.pointer_protocol import PointerProtocol
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker
from syft_proto.execution.v1.protocol_pb2 import Protocol as ProtocolPB


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
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):
        owner = owner or sy.framework.hook.local_worker
        super(Protocol, self).__init__(id, owner, tags, description, child=None)

        self.plans = plans or list()
        self.workers_resolved = len(self.plans) and all(
            isinstance(w, AbstractWorker) for w, p in self.plans
        )
        self.location: Optional[BaseWorker] = None

    def deploy(self, *workers: BaseWorker) -> "Protocol":
        """
        Calling .deploy() sends the plans to the designated workers.

        This is done in 2 phases: first, we map the fictive workers provided at creation
        (named by strings) to the provided workers, and second, we send the corresponding
        plans to each of them.
        For the first phase, either there is exactly one real worker per plan or one real
        worker per fictive_worker. _resolve_workers replaces the fictive workers by the
        real ones.

        Args:
            workers: BaseWorker. The workers to which plans are to
                be sent

        Returns:
            "Protocol": self

        Raises:
            RuntimeError: If protocol is already deployed OR
                the number of workers provided does not equal the number of fictive workers
                and does not equal the number of plans
        """
        if self.workers_resolved:
            raise RuntimeError(
                f"This protocol is already deployed to {', '.join(worker.id for worker, plan in self.plans)}."
            )

        n_workers = len(set(worker for worker, plan in self.plans))

        # to correctly map workers, we must have exactly 1 worker for 1 plan
        # or 1 worker for 1 fictive worker.
        # If we don't, raise here
        if len(self.plans) != len(workers) != n_workers:
            raise RuntimeError(
                f"This protocol is designed for {n_workers} workers, but {len(workers)} were provided."
            )

        self._resolve_workers(workers)

        # update plans list with *pointers* to the plans
        self.plans = [(worker, plan.send(worker)) for (worker, plan) in self.plans]

        return self

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        """
        Run the protocol by executing the plans sequentially

        The input args provided are sent to the first plan location. This first plan is
        run and its output is moved to the second plan location, and so on. The final
        result is returned after all plans have run, and it is composed of pointers to
        the last plan location.

        Raises:
            RuntimeError: If the protocol has a location attribute and it is not
                the local worker
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

    def request_remote_run(
        self, location: AbstractWorker, args, kwargs
    ) -> Union[List[PointerTensor], PointerTensor]:
        """
        Requests protocol execution.

        Send a request to execute the protocol on the remote location.

        Args:
            location: to which worker the request should be sent
            args: Arguments used as input data for the protocol.
            kwargs: Named arguments used as input data for the protocol.

        Returns:
            PointerTensor or list of PointerTensors: response from request to
                execute protocol
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
    def find_args_location(args) -> BaseWorker:
        """
        Return location if args contain pointers else the local worker

        Returns:
            BaseWorker: The location of a pointer if in args, else local
                worker
        """
        for arg in args:
            if isinstance(arg, FrameworkTensor):
                if hasattr(arg, "child") and isinstance(arg.child, PointerTensor):
                    return arg.location
        return sy.framework.hook.local_worker

    def send(self, location: BaseWorker) -> None:
        """
        Send a protocol to a worker, to be fetched by other workers

        Args:
            location: BaseWorker. The location to which a protocol
                is to be sent
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
    def simplify(worker: BaseWorker, protocol: "Protocol") -> Tuple:
        """
        This function takes the attributes of a Protocol and saves them in a tuple

        Args:
            worker (BaseWorker) : the worker doing the serialization
            protocol (Protocol): a Protocol object

        Returns:
            tuple: a tuple holding the unique attributes of the Protocol object

        Raises:
            TypeError: if a plan is not sy.Plan or sy.PointerPlan
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
            sy.serde.msgpack.serde._simplify(worker, protocol.id),
            sy.serde.msgpack.serde._simplify(worker, protocol.tags),
            sy.serde.msgpack.serde._simplify(worker, protocol.description),
            sy.serde.msgpack.serde._simplify(worker, plans_reference),
            sy.serde.msgpack.serde._simplify(worker, protocol.workers_resolved),
        )

    @staticmethod
    def create_from_attributes(
        worker, id, tags, description, workers_resolved, plans_assignments
    ) -> "Protocol":
        """
        This function reconstructs a Protocol object given its attributes.

        Args:
            worker: the worker doing the deserialization
            id: Protocol id
            tags: Protocol tags
            description: Protocol description
            workers_resolved: Flag whether workers are resolved
            plans_assignments: List of workers/plans IDs

        Returns:
            protocol: a Protocol object
        """
        plans = []
        for owner_id, plan_id in plans_assignments:
            if workers_resolved:
                plan_owner = worker.get_worker(owner_id, fail_hard=True)
                plan_pointer = worker.request_search(plan_id, location=plan_owner)[0]
                worker.register_obj(plan_pointer)
                plans.append((plan_owner, plan_pointer))
            else:
                try:
                    plan_owner = worker.get_worker(owner_id, fail_hard=True)
                except WorkerNotFoundException:
                    plan = worker.get_obj(plan_id)
                else:
                    plan_pointer = worker.request_search(plan_id, location=plan_owner)[0]
                    plan = plan_pointer.get()
                plans.append((worker.id, plan))

        protocol = sy.Protocol(plans=plans, id=id, owner=worker, tags=tags, description=description)

        return protocol

    @staticmethod
    def detail(worker: BaseWorker, protocol_tuple: Tuple) -> "Protocol":
        """
        This function reconstructs a Protocol object given its attributes in the form of a tuple.

        Args:
            worker: the worker doing the deserialization
            protocol_tuple: a tuple holding the attributes of the Protocol

        Returns:
            protocol: a Protocol object
        """
        id, tags, description, plans_reference, workers_resolved = map(
            lambda o: sy.serde.msgpack.serde._detail(worker, o), protocol_tuple
        )

        return Protocol.create_from_attributes(
            worker, id, tags, description, workers_resolved, plans_reference
        )

    @staticmethod
    def bufferize(worker: BaseWorker, protocol: "Protocol") -> ProtocolPB:
        """
        This function takes the attributes of a Protocol and saves them in protobuf ProtocolPB

        Args:
            worker (BaseWorker) : the worker doing the serialization
            protocol (Protocol): a Protocol object

        Returns:
            ProtocolPB: a protobuf object holding the unique attributes of the Protocol object

        Raises:
            TypeError: if a plan is not sy.Plan or sy.PointerPlan
        """
        pb_protocol = ProtocolPB()
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

            plan_assignment = pb_protocol.plan_assignments.add()
            sy.serde.protobuf.proto.set_protobuf_id(plan_assignment.worker_id, worker_id)
            sy.serde.protobuf.proto.set_protobuf_id(plan_assignment.plan_id, plan_id)

        sy.serde.protobuf.proto.set_protobuf_id(pb_protocol.id, protocol.id)
        if protocol.tags:
            pb_protocol.tags.extend(protocol.tags)
        if protocol.description:
            pb_protocol.description = protocol.description
        pb_protocol.workers_resolved = protocol.workers_resolved
        return pb_protocol

    @staticmethod
    def unbufferize(worker: AbstractWorker, pb_protocol: ProtocolPB) -> "Protocol":
        """
        This function reconstructs a Protocol object given protobuf object.

        Args:
            worker: the worker doing the deserialization
            pb_protocol: a ProtocolPB object

        Returns:
            protocol: a Protocol object
        """
        id = sy.serde.protobuf.proto.get_protobuf_id(pb_protocol.id)
        tags = set(pb_protocol.tags)
        description = pb_protocol.description
        workers_resolved = pb_protocol.workers_resolved
        plans_assignments = [
            (
                sy.serde.protobuf.proto.get_protobuf_id(item.worker_id),
                sy.serde.protobuf.proto.get_protobuf_id(item.plan_id),
            )
            for item in pb_protocol.plan_assignments
        ]

        return Protocol.create_from_attributes(
            worker, id, tags, description, workers_resolved, plans_assignments
        )

    def create_pointer(
        self, owner: AbstractWorker, garbage_collect_data: bool, tags: Set = None
    ) -> PointerProtocol:
        """
        Create a pointer to the protocol

        Args:
            owner: the owner of the pointer
            garbage_collect_data: bool
            tags: the tags inherited from the Protocol

        Returns:
            PointerProtocol: pointer to the protocol
        """
        return PointerProtocol(
            location=self.owner,
            id_at_location=self.id,
            owner=owner,
            garbage_collect_data=garbage_collect_data,
            tags=tags,
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

    def _assert_is_resolved(self) -> None:
        """
        Check if protocol has already been resolved

        Raises:
            RuntimeError: If protocol has already been resolved
        """
        if not self.workers_resolved:
            raise RuntimeError(
                "Plans have not been allocated to existing workers. Call deploy(*workers) to do so."
            )

    def _resolve_workers(self, workers: Tuple[BaseWorker, ...]) -> None:
        """
        Map the abstract workers (named by strings) to the provided workers and
        update the plans accordingly

        Args:
            workers: Iterable of workers. The workers to map to workers
                in the protocol
        """
        dict_workers = {w.id: w for w in workers}
        set_fake_ids = set(worker for worker, _ in self.plans)
        set_real_ids = set(dict_workers.keys())

        if 0 < len(set_fake_ids.intersection(set_real_ids)) < len(set_real_ids):
            # The user chose fake ids that correspond to real ids but not all of them match.
            # Maybe it's a mistake so we warn the user.
            warnings.warn(
                "You are deploying a protocol with workers for which only a subpart"
                "have ids that match an id chosen for the protocol."
            )

        # If the "fake" ids manually set by the user when writing the protocol exactly match the ids
        # of the workers, these fake ids in self.plans are replaced with the real workers.
        if set_fake_ids == set_real_ids:
            self.plans = [(dict_workers[w], p) for w, p in self.plans]

        # If there is an exact one-to-one mapping, just iterate and keep the order
        # provided when assigning the workers
        elif len(workers) == len(self.plans):
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
