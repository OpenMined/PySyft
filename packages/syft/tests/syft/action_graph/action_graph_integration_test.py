"""
Tests for the integration of action graph into action and action object creation / update
"""

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
import syft as sy
from syft.client.client import SyftClient
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject

# relative
from ..service.action.action_object_test import helper_make_action_obj
from ..service.action.action_object_test import helper_make_action_pointers


@pytest.mark.parametrize(
    "num_assets",
    [1, 2, 3],
)
def test_node_creation_dataset_upload(root_client: SyftClient, num_assets: int) -> None:
    """
    Create a node in the graph when a dataset is uploaded
    """
    # let's upload a dataset with #num_assets assets inside of it
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


def test_node_creation_action_obj_send(root_client: SyftClient) -> None:
    """
    Test the creation of a node in the graph when the action_obj.send method is called
    """
    action_obj_a = ActionObject.from_obj([2, 4, 6])
    action_obj_a.send(root_client)
    action_obj_b = ActionObject.from_obj([1, 2, 3])
    action_obj_b.send(root_client)
    assert len(root_client.api.services.graph.nodes()) == 2
    assert len(root_client.api.services.graph.edges()) == 0


def test_action_graph_creation_no_mutation(root_client: SyftClient) -> None:
    """
    Test the creation of an action graph when we do these non-mutating operations:

    a = root_domain_client.api.lib.numpy.array([1,2,3])
    b = root_domain_client.api.lib.numpy.array([2,3,4])
    c = a + b
    d = root_domain_client.api.lib.numpy.array([1, 2, 3])
    e = c * d
    """
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


def test_node_creation_generate_remote_lib_function(root_client: SyftClient) -> None:
    """
    Create a graph node when an Action is
    created when a client calls generate_remote_lib_function
    """
    a = root_client.api.lib.numpy.array([1, 2, 3])
    b = root_client.api.lib.numpy.array([2, 3, 4])
    c = root_client.api.lib.numpy.add(a, b)
    d = root_client.api.lib.numpy.array([3, 4, 5])
    root_client.api.lib.numpy.multiply(c, d)
    assert len(root_client.api.services.graph.nodes()) == 13
    assert len(root_client.api.services.graph.edges()) == 12


@pytest.mark.skip
def test_node_creation_syft_make_action(root_client: SyftClient) -> None:
    """
    Create a graph node when an Action is
    created in the syft_make_action method of ActionObject
    """
    orig_obj, op, args, kwargs = (int(1), "__add__", [1], {})
    obj_id = Action.make_id(None)
    lin_obj_id = Action.make_result_id(obj_id)
    obj = ActionObject.from_obj(orig_obj, id=obj_id, syft_lineage_id=lin_obj_id)
    root_client.api.services.action.set(obj)
    obj_pointer = root_client.api.services.action.get_pointer(obj.id)
    obj_pointer.syft_make_action(str(type(orig_obj)), op, args=args, kwargs=kwargs)

    print(helper_make_action_obj)
    print(helper_make_action_pointers)
