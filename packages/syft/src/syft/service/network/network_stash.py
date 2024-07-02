# third party
from result import Result

# relative
from ...abstract_node import NodeType
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...util.telemetry import instrument
from ..data_subject.data_subject import NamePartitionKey
from ..response import SyftError
from .node_peer import NodePeer
from .node_peer import NodePeerUpdate

VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)
NodeTypePartitionKey = PartitionKey(key="node_type", type_=NodeType)
OrderByNamePartitionKey = PartitionKey(key="name", type_=str)


@instrument
@serializable()
class NetworkStash(BaseUIDStoreStash):
    object_type = NodePeer
    settings: PartitionSettings = PartitionSettings(
        name=NodePeer.__canonical_name__, object_type=NodePeer
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[NodePeer | None, str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(
        self,
        credentials: SyftVerifyKey,
        peer_update: NodePeerUpdate,
        has_permission: bool = False,
    ) -> Result[NodePeer, str]:
        valid = self.check_type(peer_update, NodePeerUpdate)
        if valid.is_err():
            return SyftError(message=valid.err())
        return super().update(credentials, peer_update, has_permission=has_permission)

    def create_or_update_peer(
        self, credentials: SyftVerifyKey, peer: NodePeer
    ) -> Result[NodePeer, str]:
        """
        Update the selected peer and its route priorities if the peer already exists
        If the peer does not exist, simply adds it to the database.

        Args:
            credentials (SyftVerifyKey): The credentials used to authenticate the request.
            peer (NodePeer): The peer to be updated or added.

        Returns:
            Result[NodePeer, str]: The updated or added peer if the operation
            was successful, or an error message if the operation failed.
        """
        valid = self.check_type(peer, NodePeer)
        if valid.is_err():
            return SyftError(message=valid.err())

        existing = self.get_by_uid(credentials=credentials, uid=peer.id)
        if existing.is_ok() and existing.ok() is not None:
            existing_peer: NodePeer = existing.ok()
            existing_peer.update_routes(peer.node_routes)
            peer_update = NodePeerUpdate(
                id=peer.id, node_routes=existing_peer.node_routes
            )
            result = self.update(credentials, peer_update)
            return result
        else:
            result = self.set(credentials, peer)
            return result

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[NodePeer | None, SyftError]:
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_one(credentials, qks)

    def get_by_node_type(
        self, credentials: SyftVerifyKey, node_type: NodeType
    ) -> Result[list[NodePeer], SyftError]:
        qks = QueryKeys(qks=[NodeTypePartitionKey.with_obj(node_type)])
        return self.query_all(
            credentials=credentials, qks=qks, order_by=OrderByNamePartitionKey
        )
