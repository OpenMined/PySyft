import syft as sy

from syft.generic.object import AbstractObject

from typing import List


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
        super(Protocol, self).__init__(id, owner, tags, description, child)
        self.plans = plans if plans else list()

    def __repr__(self):
        repr = "Protocol"
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

        worker_map = {
            abstract_worker: worker
            for worker, abstract_worker in zip(
                workers, set(abstract_worker for abstract_worker, plan in self.plans)
            )
        }

        self.plans = [(worker_map[abstract_worker], plan) for abstract_worker, plan in self.plans]

        for worker, plan in self.plans:
            plan.send(worker)

    def run(self, *args, synchronous=True):
        """
        Run the protocol bu executing the plans

        In synchronous mode, the input args provided is sent to the first plan
        location. This first plan is run and its output is moved to the second
        plan location, and so on. The final result is returned after all plans
        have run.
        """
        previous_worker_id = None
        for worker, plan in self.plans:
            # Transmit the args to the next worker if it's a different one % the previous
            if previous_worker_id is not None:
                if previous_worker_id != worker.id:
                    print("move", previous_worker_id, " -> ", worker.id)
                    args = [arg.move(worker) for arg in args]
            else:
                print("send", worker.id)
                args = [arg.send(worker) for arg in args]

            previous_worker_id = worker.id

            print("args", args)
            response = plan(*args)
            print("response", response)

            args = response if isinstance(response, tuple) else (response,)

        return response

    def send(self, location):
        """
        Send a protocol to a worker, to be fetched by other workers
        """
        raise NotImplementedError
