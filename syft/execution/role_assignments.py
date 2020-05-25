from syft.workers.abstract import AbstractWorker


class RoleAssignments:
    """ This object is basically a map from role ids to workers.

    It is used in Protocol execution.
    The RoleAssignment associated with a Protocol will be sent to each worker
    having joined the Protocol before an execution so that each party can know
    which other parties are participating and communicate with them if needed.
    """

    def __init__(self, role_ids):
        """
        Args:
            role_ids: an iterable containing values that indentify the roles of
                the Protocol to which the RoleAssignments is associated.
        """
        self.assignments = {role_id: [] for role_id in role_ids}

    def assign(self, role_id, worker):
        """ Assign a specific worker to the specified role.
        """
        if role_id not in self.assignments:
            raise ValueError(
                f"role_id {role_id} not present in RoleAssignments "
                f"with roles {', '.join(self.assignments.keys())}"
            )

        if isinstance(worker, list):
            self.assignments[role_id].extend(worker)
        else:
            self.assignments[role_id].append(worker)

    def unassign(self, role_id, worker):
        """ Unassign a specific worker from the specified role.
        """
        if role_id not in self.assignments:
            raise ValueError(
                f"role_id {role_id} not present in RoleAssignments "
                f"with roles {', '.join(self.assignments.keys())}"
            )
        self.assignments[role_id].remove(worker)

    def reset(self):
        """ Remove all the workers from the assignment dict.
        """
        self.assignments = {role_id: [] for role_id in self.assignments}

    @staticmethod
    def simplify(worker: AbstractWorker, assignments: "RoleAssignments") -> tuple:
        """ Simplify a RoleAssignments object.
        """
        pass

    @staticmethod
    def detail(worker: AbstractWorker, simplified_assignments: tuple) -> "RoleAssignments":
        """ Detail a simplified RoleAssignments.
        """
        pass
