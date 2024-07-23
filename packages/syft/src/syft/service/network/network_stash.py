# third party
from result import Result

# relative
from ...abstract_server import ServerType
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...util.telemetry import instrument
from ..data_subject.data_subject import NamePartitionKey
from ..response import SyftError
from .server_peer import ServerPeer
from .server_peer import ServerPeerUpdate

VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)
ServerTypePartitionKey = PartitionKey(key="server_type", type_=ServerType)
OrderByNamePartitionKey = PartitionKey(key="name", type_=str)


@instrument
@serializable(canonical_name="NetworkStash", version=1)
class NetworkStash(BaseUIDStoreStash):
    object_type = ServerPeer
    settings: PartitionSettings = PartitionSettings(
        name=ServerPeer.__canonical_name__, object_type=ServerPeer
    )

    def __init__(self, store: DocumentStore) -> None:
        super().__init__(store=store)

    def get_by_name(
        self, credentials: SyftVerifyKey, name: str
    ) -> Result[ServerPeer | None, str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(
        self,
        credentials: SyftVerifyKey,
        peer_update: ServerPeerUpdate,
        has_permission: bool = False,
    ) -> Result[ServerPeer, str]:
        valid = self.check_type(peer_update, ServerPeerUpdate)
        if valid.is_err():
            return SyftError(message=valid.err())
        return super().update(credentials, peer_update, has_permission=has_permission)

    def create_or_update_peer(
        self, credentials: SyftVerifyKey, peer: ServerPeer
    ) -> Result[ServerPeer, str]:
        """
        Update the selected peer and its route priorities if the peer already exists
        If the peer does not exist, simply adds it to the database.

        Args:
            credentials (SyftVerifyKey): The credentials used to authenticate the request.
            peer (ServerPeer): The peer to be updated or added.

        Returns:
            Result[ServerPeer, str]: The updated or added peer if the operation
            was successful, or an error message if the operation failed.
        """
        valid = self.check_type(peer, ServerPeer)
        if valid.is_err():
            return SyftError(message=valid.err())

        existing = self.get_by_uid(credentials=credentials, uid=peer.id)
        if existing.is_ok() and existing.ok() is not None:
            existing_peer: ServerPeer = existing.ok()
            existing_peer.update_routes(peer.server_routes)
            peer_update = ServerPeerUpdate(
                id=peer.id, server_routes=existing_peer.server_routes
            )
            result = self.update(credentials, peer_update)
            return result
        else:
            result = self.set(credentials, peer)
            return result

    def get_by_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[ServerPeer | None, SyftError]:
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_one(credentials, qks)

    def get_by_server_type(
        self, credentials: SyftVerifyKey, server_type: ServerType
    ) -> Result[list[ServerPeer], SyftError]:
        qks = QueryKeys(qks=[ServerTypePartitionKey.with_obj(server_type)])
        return self.query_all(
            credentials=credentials, qks=qks, order_by=OrderByNamePartitionKey
        )
