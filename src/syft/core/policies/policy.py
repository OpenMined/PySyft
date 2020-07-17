class Policy:
    """Policy is the interface for polices. A policy is a constraint on a
    worker. Constraints can be made on memory usage, cpu load, message
    processing frequency, etc.
    """

    def raise_error(self) -> None:
        """Method that raises a specific error when the policy is violated."""
        raise NotImplementedError

    def enforce_policy(self, worker: "Worker") -> None:
        """Method that checks if a policy is violated by a worker or not.

        Args:
            worker: the worker on which to enforce the policy.
        """