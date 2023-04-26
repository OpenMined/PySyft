# stdlib

# third party

# syft absolute
from syft.node.credentials import SyftVerifyKey
from syft.service.action.action_graph import InMemoryActionGraphStore
from syft.service.action.action_graph import InMemoryGraphConfig
from syft.service.action.action_graph import NodeActionData
from syft.service.action.action_object import Action
from syft.service.action.action_object import ActionObject


def test_in_memory_action_graph_store_serde(
    in_mem_graph_config: InMemoryGraphConfig,
    verify_key: SyftVerifyKey,
) -> None:
    """
    Create an InMemoryActionGraphStore, add some nodes and edges to it
    serialize it, deserialize it, then check if the deserialized
    InMemoryActionGraphStore object is equal to the original one.
    """
    graph_store = InMemoryActionGraphStore(store_config=in_mem_graph_config)
    # create some actions and add them to the graph store
    action_obj_a = ActionObject.from_obj([2, 4, 6])
    action_obj_b = ActionObject.from_obj([2, 3, 4])
    # action1 -> a + b = c
    action1 = Action(
        path="action.execute",
        op="__add__",
        remote_self=action_obj_a.syft_lineage_id,
        args=[action_obj_b.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action1)
    # action2 -> initialization of variable d
    action_obj_d = ActionObject.from_obj([1, 2, 3])
    action2 = Action(
        path="action.execute",
        op="np.array",
        remote_self=None,
        args=[action_obj_d.syft_lineage_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action2)
    # action3 -> c * d
    action3 = Action(
        path="action.execute",
        op="__mul__",
        remote_self=action1.result_id,
        args=[action2.result_id],
        kwargs={},
    )
    graph_store.set(credentials=verify_key, action=action3)
    # action4 -> d.astype('int32')
    as_type_action_obj = ActionObject.from_obj("np.int32")
    action4 = Action(
        path="action.execute",
        op="astype",
        remote_self=action2.result_id,
        args=[as_type_action_obj.syft_lineage_id],
        kwargs={},
        result_id=action2.result_id,
    )
    graph_store.set(credentials=verify_key, action=action4)

    # check if the node action 2 has been mutated
    node_action_data_2: NodeActionData = graph_store.get(
        uid=action2.id, credentials=verify_key
    ).ok()
    assert node_action_data_2.is_mutated is True

    # serializing and deserializing the graph store
