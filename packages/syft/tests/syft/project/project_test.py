# third party
from pydantic import ValidationError
import pytest

# syft absolute
import syft as sy
from syft.service.project.project import Project


def test_project_creation(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon",
        email="sheldon@caltech.edu",
        password="bazinga",
        password_verify="bazinga",
    )

    ds_client = sy.login(server=worker, email="sheldon@caltech.edu", password="bazinga")

    new_project = sy.Project(
        name="My Cool Project", description="My Cool Description", members=[ds_client]
    )

    project = new_project.send()

    assert isinstance(project, Project)
    assert new_project.id == project.id
    assert project.members[0].verify_key == root_client.verify_key
    assert project.users[0].verify_key == ds_client.verify_key
    assert project.name == "My Cool Project"
    assert project.description == "My Cool Description"


def test_data_owner_project_creation(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon",
        email="sheldon@caltech.edu",
        password="bazinga",
        password_verify="bazinga",
    )

    new_project = sy.Project(
        name="My Cool Project", description="My Cool Description", members=[root_client]
    )

    project = new_project.send()
    assert project.name == "My Cool Project"


def test_exception_different_email(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon",
        email="sheldon@caltech.edu",
        password="bazinga",
        password_verify="bazinga",
    )

    root_client.register(
        name="leonard",
        email="leonard@princeton.edu",
        password="penny",
        password_verify="penny",
    )

    ds_sheldon = sy.login(
        server=worker, email="sheldon@caltech.edu", password="bazinga"
    )

    ds_leonard = sy.login(
        server=worker, email="leonard@princeton.edu", password="penny"
    )

    with pytest.raises(ValidationError):
        sy.Project(
            name="My Cool Project",
            description="My Cool Description",
            members=[ds_sheldon, ds_leonard],
        )


def test_project_serde(worker):
    root_client = worker.root_client

    root_client.register(
        name="sheldon",
        email="sheldon@caltech.edu",
        password="bazinga",
        password_verify="bazinga",
    )

    ds_sheldon = sy.login(
        server=worker, email="sheldon@caltech.edu", password="bazinga"
    )

    new_project = sy.Project(
        name="My Cool Project", description="My Cool Description", members=[ds_sheldon]
    )

    project = new_project.send()

    ser_data = sy.serialize(project, to_bytes=True)
    assert isinstance(ser_data, bytes)

    deser_data = sy.deserialize(ser_data, from_bytes=True)
    assert isinstance(deser_data, type(project))
    assert deser_data == project
