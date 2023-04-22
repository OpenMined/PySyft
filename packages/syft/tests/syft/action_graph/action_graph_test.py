"""
Tests for the classes in the syft.service.action.action_graph module:
    - NodeActionData, NodeActionDataUpdate
    - InMemoryStoreClientConfig
    - NetworkXBackingStore
    - InMemoryActionGraphStore
"""

# third party

# syft absolute
from syft.node.credentials import SyftSigningKey
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import ActionStatus
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject
from syft.service.context import AuthedServiceContext


def test_node_action_data() -> None:
    # credentials needed to create a NodeActionData
    signing_key = SyftSigningKey.generate()
    verify_key: SyftVerifyKey = signing_key.verify_key
    authed_context = AuthedServiceContext(credentials=verify_key)
    assert authed_context.credentials == verify_key

    # create a NodeActionData from an Action
    action_obj = ActionObject.from_obj([1, 2, 3])
    action = Action(
        path="action.execute",
        op="np.array",
        args=[action_obj.syft_lineage_id],
        kwargs={},
    )
    node_action_data = NodeActionData.from_action(action=action, credentials=verify_key)

    assert node_action_data.id == action.id
    assert node_action_data.action == action
    assert node_action_data.user_verify_key == verify_key
    assert node_action_data.status == ActionStatus.PROCESSING
    assert node_action_data.__hash__() == node_action_data.action.syft_history_hash

    node_action_data_duplicate = NodeActionData.from_action(
        action=action, credentials=verify_key
    )
    assert node_action_data_duplicate == node_action_data


# @pytest.mark.parametrize("graph_client", [InMemoryGraphClient])
# def test_graph_creation(graph_client):
#     action_graph = ActionGraph(node_uid=UID(), graph_client=graph_client)
#     assert isinstance(action_graph, ActionGraph)
#     assert isinstance(action_graph.client, graph_client)
#     assert isinstance(action_graph.client.graph, graph_client.graph_type)


# @pytest.mark.parametrize("graph_client", [InMemoryGraphClient])
# def test_node_creation(worker, graph_client):
#     action_graph = ActionGraph(node_uid=worker.id, graph_client=graph_client)
#     assert action_graph

#     action_obj = ActionObject.from_obj([1, 2, 3])
#     assert action_obj

#     action = Action(
#         path="action.execute",
#         op="np.array",
#         args=[action_obj.syft_lineage_id],
#         kwargs={},
#     )
#     assert action
#     action_graph.add_action(action)

#     node = NodeActionData.from_action(action)

#     assert node in action_graph.client.graph.nodes
#     assert action_graph.client.graph.number_of_nodes() == 1
#     assert action_graph.client.graph.number_of_edges() == 0


# @pytest.mark.parametrize("graph_client", [InMemoryGraphClient])
# def test_edge_creation(worker, graph_client):
#     action_graph = ActionGraph(node_uid=worker.id, graph_client=graph_client)
#     assert action_graph

#     action_obj_a = ActionObject.from_obj([1, 2, 3])
#     action_obj_b = ActionObject.from_obj([2, 4, 5])

#     # First action -> np.array([1, 2, 3])
#     # Create a numpy array from action_obj_a
#     action_a = Action(
#         path="action.execute",
#         op="np.array",
#         args=[action_obj_a.syft_lineage_id],
#         kwargs={},
#     )
#     action_graph.add_action(action_a)
#     node = NodeActionData.from_action(action_a)
#     assert node in action_graph.client.graph.nodes

#     # Second action -> np.array([2, 4, 5])
#     # Create a numpy array from action_obj_b
#     action_b = Action(
#         path="action.execute",
#         op="np.array",
#         args=[action_obj_b.syft_lineage_id],
#         kwargs={},
#     )
#     action_graph.add_action(action_b)

#     node = NodeActionData.from_action(action_b)
#     assert node in action_graph.client.graph.nodes

#     # Third action -> action_obj_a + action_obj_b
#     # Add two arrays
#     action_add = Action(
#         path="action.execute",
#         op="__add__",
#         remote_self=action_a.result_id,
#         args=[action_b.result_id],
#         kwargs={},
#     )
#     action_graph.add_action(action_add)
#     node = NodeActionData.from_action(action_add)

#     assert node in action_graph.client.graph.nodes
#     assert action_graph.client.graph.number_of_nodes() == 3
#     assert action_graph.client.graph.number_of_edges() == 2


# def test_node_removal(worker, graph_client):
#     pass
