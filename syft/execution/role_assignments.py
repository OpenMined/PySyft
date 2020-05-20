from syft.workers.abstract import AbstractWorker


class RoleAssignments:
    """
    """

    def __init__(self, role_ids):
        self.assignments = {role_id: [] for role_id in role_ids}

    def assign(self, role_id, worker):
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
        if role_id not in self.assignments:
            raise ValueError(
                f"role_id {role_id} not present in RoleAssignments "
                f"with roles {', '.join(self.assignments.keys())}"
            )
        self.assignments[role_id].remove(worker)

    def reset(self):
        self.assignments = {role_id: [] for role_id in self.assignments}

    @staticmethod
    def simplify(worker: AbstractWorker, assignments: "RoleAssignments") -> tuple:
        pass

    @staticmethod
    def detail(worker: AbstractWorker, simplified_assignments: tuple) -> "RoleAssignments":
        pass
