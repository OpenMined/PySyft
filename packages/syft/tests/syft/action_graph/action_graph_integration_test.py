"""
Tests for the integration of action graph into action and action object creation / update
"""

# third party
import pytest

# syft absolute
from syft.client.client import SyftClient


def test_action_graph_creation_no_mutation(root_domain_client: SyftClient) -> None:
    pass


@pytest.mark.skip
def test_node_creation_dataset_upload(root_domain_client: SyftClient) -> None:
    """
    Create a node in the graph when a dataset is uploaded
    """
    pass


@pytest.mark.skip
def test_node_creation_action_obj_send() -> None:
    """
    Create a node in the graph when the action_obj.send method is called
    """
    pass


@pytest.mark.skip
def test_node_creation_generate_remote_lib_function() -> None:
    """
    Create a graph node when an Action is
    created when a client generate a remote function
    """
    pass


@pytest.mark.skip
def test_node_creation_syft_make_action() -> None:
    """
    Create a graph node when an Action is
    created in the syft_make_action method of ActionObject
    """
    pass
