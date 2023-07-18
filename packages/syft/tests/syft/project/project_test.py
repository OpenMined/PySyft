# third party
import pytest

# syft absolute
import syft as sy
from syft.service.project.project import Project


def test_project_creation(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon", email="sheldon@caltech.edu", password="bazinga"
    )

    ds_client = sy.login(node=worker, email="sheldon@caltech.edu", password="bazinga")

    new_project = sy.Project(
        name="My Cool Project", description="My Cool Description", members=[ds_client]
    )

    project = new_project.start()

    assert isinstance(project, Project)
    assert new_project.id == project.id
    assert project.members[0].verify_key == root_client.verify_key
    assert project.users[0].verify_key == ds_client.verify_key
    assert project.name == "My Cool Project"
    assert project.description == "My Cool Description"


def test_error_data_owner_project_creation(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon", email="sheldon@caltech.edu", password="bazinga"
    )

    new_project = sy.Project(
        name="My Cool Project", description="My Cool Description", members=[root_client]
    )

    project = new_project.start()

    assert isinstance(project, sy.SyftError)
    assert project.message == "Only Data Scientists can create projects"


def test_exception_different_email(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon", email="sheldon@caltech.edu", password="bazinga"
    )

    root_client.register(
        name="leonard", email="leonard@princeton.edu", password="penny"
    )

    ds_sheldon = sy.login(node=worker, email="sheldon@caltech.edu", password="bazinga")

    ds_leonard = sy.login(
        node=worker, email="leonard@princeton.edu", password="starwars"
    )

    with pytest.raises(Exception):
        sy.Project(
            name="My Cool Project",
            description="My Cool Description",
            members=[ds_sheldon, ds_leonard],
        )

def test_project_serde(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon", email="sheldon@caltech.edu", password="bazinga"
    )

    new_project = sy.Project(
        name="My Cool Project", description="My Cool Description", members=[root_client]
    )

    project = new_project.start()

    ser_data = sy.serialize(project, to_bytes=True)
    assert isinstance(ser_data, bytes)

    deser_data = sy.deserialize(ser_data, from_bytes=True)
    assert isinstance(deser_data, type(project))
    assert deser_data == project