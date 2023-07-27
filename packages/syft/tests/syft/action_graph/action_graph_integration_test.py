"""
Tests for the integration of action graph into action and action object creation / update
"""

# stdlib
import uuid

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
import syft as sy
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject


@pytest.mark.parametrize(
    "num_assets",
    [1, 2, 3],
)
def test_node_creation_dataset_upload(num_assets: int) -> None:
    """
    Create a node in the graph when a dataset is uploaded
    """
    # let's upload a dataset with #num_assets assets inside of it
    name = uuid.uuid4().hex
    worker = sy.Worker.named(name=name, reset=True)
    root_client = worker.root_client
    dataset = sy.Dataset(name="Test Dataset")
    dataset.set_description("""Test Dataset""")
    dataset.add_citation("Person, place or thing")
    country = sy.DataSubject(name="Country", aliases=["country_code"])
    canada = sy.DataSubject(name="Canada", aliases=["country_code:ca"])
    country.add_member(canada)
    registry = root_client.data_subject_registry
    registry.add_data_subject(country)
    for i in range(num_assets):
        data = pd.DataFrame(
            np.random.randint(0, 100, size=(10, 4)), columns=list("ABCD")
        )
        mock = pd.DataFrame(
            np.random.randint(0, 100, size=(10, 4)), columns=list("ABCD")
        )
        ctf = sy.Asset(name=f"test_dataset_{i}")
        ctf.set_description("""all the datas""")
        ctf.set_obj(data)
        ctf.set_shape((10, 4))
        ctf.add_data_subject(canada)
        ctf.set_mock(mock, mock_is_real=False)
        dataset.add_asset(ctf)
    root_client.upload_dataset(dataset)
    # each asset will be a node in our action graph
    assert len(root_client.api.services.graph.nodes()) == num_assets
    assert len(root_client.api.services.graph.edges()) == 0


def test_node_creation_action_obj_send() -> None:
    """
    Test the creation of a node in the graph when the action_obj.send method is called
    """
    name = uuid.uuid4().hex
    worker = sy.Worker.named(name=name, reset=True)
    root_client = worker.root_client
    action_obj_a = ActionObject.from_obj([2, 4, 6])
    action_obj_a.send(root_client)
    action_obj_b = ActionObject.from_obj([1, 2, 3])
    action_obj_b.send(root_client)
    assert len(root_client.api.services.graph.nodes()) == 2
    assert len(root_client.api.services.graph.edges()) == 0


def test_action_graph_creation_no_mutation() -> None:
    """
    Test the creation of an action graph when we do these non-mutating operations:

    a = root_domain_client.api.lib.numpy.array([1,2,3])
    b = root_domain_client.api.lib.numpy.array([2,3,4])
    c = a + b
    d = root_domain_client.api.lib.numpy.array([1, 2, 3])
    e = c * d
    """
    name = uuid.uuid4().hex
    worker = sy.Worker.named(name=name, reset=True)
    root_client = worker.root_client
    a = root_client.api.lib.numpy.array([1, 2, 3])
    b = root_client.api.lib.numpy.array([2, 3, 4])
    assert len(root_client.api.services.graph.nodes()) == 6
    assert len(root_client.api.services.graph.edges()) == 4
    c = a + b
    assert len(root_client.api.services.graph.nodes()) == 8
    assert len(root_client.api.services.graph.edges()) == 7
    d = root_client.api.lib.numpy.array([3, 4, 5])
    assert len(root_client.api.services.graph.nodes()) == 11
    assert len(root_client.api.services.graph.edges()) == 9
    c * d
    assert len(root_client.api.services.graph.nodes()) == 13
    assert len(root_client.api.services.graph.edges()) == 12


def test_node_creation_generate_remote_lib_function() -> None:
    """
    Create a graph node when an Action is
    created when a client calls generate_remote_lib_function
    """
    name = uuid.uuid4().hex
    worker = sy.Worker.named(name=name, reset=True)
    root_client = worker.root_client
    a = root_client.api.lib.numpy.array([1, 2, 3])  # 3 nodes
    b = root_client.api.lib.numpy.array([2, 3, 4])  # 3 nodes
    c = root_client.api.lib.numpy.add(a, b)  # 2 nodes
    d = root_client.api.lib.numpy.array([3, 4, 5])  # 3 nodes
    root_client.api.lib.numpy.multiply(c, d)  # 2 nodes
    assert len(root_client.api.services.graph.nodes()) == 13
    assert len(root_client.api.services.graph.edges()) == 12


@pytest.mark.parametrize(
    "testcase",
    [
        # (object, operation, *args, **kwargs)
        ("abc", "find", ["b"], {}),
        (int(1), "__add__", [1], {}),
        (float(1.2), "__add__", [1], {}),
        ((1, 1, 3), "count", [1], {}),
    ],
)
def test_node_creation_syft_make_action(testcase) -> None:
    """
    Create a graph node when an Action is
    created in the syft_make_action method of ActionObject
    """
    name = uuid.uuid4().hex
    worker = sy.Worker.named(name=name, reset=True)
    root_client = worker.root_client
    orig_obj, op, args, kwargs = testcase
    obj_id = Action.make_id(None)
    lin_obj_id = Action.make_result_id(obj_id)
    obj = ActionObject.from_obj(orig_obj, id=obj_id, syft_lineage_id=lin_obj_id)
    obj_pointer = obj.send(root_client)
    path = str(type(orig_obj))
    obj_pointer.syft_make_action(
        path=path, op=op, remote_self=obj_pointer.id, args=args, kwargs=kwargs
    )

    assert len(root_client.api.services.graph.nodes()) == 5
    assert len(root_client.api.services.graph.edges()) == 4
