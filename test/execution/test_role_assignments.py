from syft.execution.role_assignments import RoleAssignments


def test_assign(workers):
    alice = workers["alice"]
    bob = workers["bob"]

    role_assignment = RoleAssignments(["role1", "role2"])

    role_assignment.assign("role1", alice)
    role_assignment.assign("role2", bob)

    assert role_assignment.assignments["role1"] == [alice]
    assert role_assignment.assignments["role2"] == [bob]


def test_unassign(workers):
    alice = workers["alice"]
    bob = workers["bob"]

    role_assignment = RoleAssignments(["role1"])

    role_assignment.assign("role1", alice)
    role_assignment.assign("role1", bob)
    role_assignment.unassign("role1", alice)

    assert role_assignment.assignments["role1"] == [bob]
