import syft as sy

from syft.generic.frameworks.types import FrameworkTensor
from syft.generic.object import AbstractObject
from syft.generic.pointers.pointer_tensor import PointerTensor
from syft.workers.abstract import AbstractWorker
from syft.workers.base import BaseWorker

from typing import List, Union


class Protocol(AbstractObject):
    def __init__(
        self,
        plans: List = None,
        id: int = None,
        owner: "sy.workers.AbstractWorker" = None,
        tags: List[str] = None,
        description: str = None,
        child=None,
    ):
        if owner is None:
            owner = sy.framework.hook.local_worker

        super(Protocol, self).__init__(id, owner, tags, description, child)
        self.plans = plans if plans else list()

        self.workers_resolved = len(self.plans) and all(
            isinstance(w, AbstractWorker) for w, p in self.plans
        )
        self.location = None

    def deploy(self, *workers):
        """
        Deploy the plans on the designated workers.

        Map the abstract workers (named by strings) to the provided workers, and
        send the corresponding plans to each of them
        """
        n_workers = len(set(abstract_worker for abstract_worker, plan in self.plans))

        if len(workers) != n_workers:
            raise RuntimeError(
                f"This protocol is designed for {n_workers} workers, but {len(workers)} were provided."
            )

        self._resolve_workers(workers)

        for worker, plan in self.plans:
            plan.send(worker)

    def run(self, *args, **kwargs):
        """
        Run the protocol bu executing the plans

        In synchronous mode, the input args provided is sent to the first plan
        location. This first plan is run and its output is moved to the second
        plan location, and so on. The final result is returned after all plans
        have run.
        """
        synchronous = kwargs.get("synchronous", True)
        self._assert_is_resolved()

        if self.location is not None:
            location = Protocol.find_args_location(args)
            if location != self.location:
                print(location, self.location)
                raise RuntimeError(
                    f"This protocol has been sent to {self.location.id}, but you provided "
                    f"local arguments or pointers to {location.id}."
                )

        if synchronous:
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

        else:
            raise NotImplementedError("Promises are not currently supported")

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
