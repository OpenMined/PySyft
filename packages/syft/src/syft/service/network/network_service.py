# stdlib
import secrets
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

# third party
from result import Result

# relative
from ...abstract_node import NodeType
from ...client.client import HTTPConnection
from ...client.client import PythonConnection
from ...client.client import SyftClient
from ...node.credentials import SyftVerifyKey
from ...node.worker_settings import WorkerSettings
from ...serde.serializable import serializable
from ...store.document_store import BaseUIDStoreStash
from ...store.document_store import DocumentStore
from ...store.document_store import PartitionKey
from ...store.document_store import PartitionSettings
from ...store.document_store import QueryKeys
from ...types.grid_url import GridURL
from ...types.transforms import TransformContext
from ...types.transforms import keep
from ...types.transforms import transform
from ...types.transforms import transform_method
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..data_subject.data_subject import NamePartitionKey
from ..metadata.node_metadata import NodeMetadata
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from ..vpn.headscale_client import HeadscaleAuthToken
from ..vpn.headscale_client import HeadscaleClient
from ..vpn.tailscale_client import TailscaleClient
from ..vpn.tailscale_client import TailscaleState
from ..vpn.tailscale_client import TailscaleStatus
from ..vpn.tailscale_client import get_vpn_client
from .node_peer import NodePeer
from .routes import HTTPNodeRoute
from .routes import NodeRoute
from .routes import PythonNodeRoute

VerifyKeyPartitionKey = PartitionKey(key="verify_key", type_=SyftVerifyKey)
NodeTypePartitionKey = PartitionKey(key="node_type", type_=NodeType)


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
    ) -> Result[Optional[NodePeer], str]:
        qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
        return self.query_one(credentials=credentials, qks=qks)

    def update(
        self, credentials: SyftVerifyKey, peer: NodePeer
    ) -> Result[NodePeer, str]:
        valid = self.check_type(peer, NodePeer)
        if valid.is_err():
            return SyftError(message=valid.err())
        return super().update(credentials, peer)

    def update_peer(
        self, credentials: SyftVerifyKey, peer: NodePeer
    ) -> Result[NodePeer, str]:
        valid = self.check_type(peer, NodePeer)
        if valid.is_err():
            return SyftError(message=valid.err())
        existing = self.get_by_uid(credentials=credentials, uid=peer.id)
        if existing.is_ok() and existing.ok():
            existing = existing.ok()
            existing.update_routes(peer.node_routes)
            result = self.update(credentials, existing)
            return result
        else:
            result = self.set(credentials, peer)
            return result

    def get_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> Result[NodePeer, SyftError]:
        qks = QueryKeys(qks=[VerifyKeyPartitionKey.with_obj(verify_key)])
        return self.query_one(credentials, qks)

    def get_by_node_type(
        self, credentials: SyftVerifyKey, node_type: NodeType
    ) -> Result[List[NodePeer], SyftError]:
        qks = QueryKeys(qks=[NodeTypePartitionKey.with_obj(node_type)])
        return self.query_all(credentials=credentials, qks=qks)


@instrument
@serializable()
class NetworkService(AbstractService):
    store: DocumentStore
    stash: NetworkStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = NetworkStash(store=store)

    # TODO: Check with MADHAVA, can we even allow guest user to introduce routes to
    # domain nodes?
    @service_method(
        path="network.exchange_credentials_with",
        name="exchange_credentials_with",
        roles=GUEST_ROLE_LEVEL,
    )
    def exchange_credentials_with(
        self,
        context: AuthedServiceContext,
        self_node_route: NodeRoute,
        remote_node_route: NodeRoute,
        remote_node_verify_key: SyftVerifyKey,
    ) -> Union[SyftSuccess, SyftError]:
        """Exchange Route With Another Node"""

        # Step 1: Validate the Route
        self_node_peer = self_node_route.validate_with_context(context=context)

        if isinstance(self_node_peer, SyftError):
            return self_node_peer

        # Step 2: Send the Node Peer to the remote node
        # Also give them their own to validate that it belongs to them
        # random challenge prevents replay attacks
        remote_client = remote_node_route.client_with_context(context=context)
        random_challenge = secrets.token_bytes(16)

        remote_res = remote_client.api.services.network.add_peer(
            peer=self_node_peer,
            challenge=random_challenge,
            self_node_route=remote_node_route,
            verify_key=remote_node_verify_key,
        )

        if isinstance(remote_res, SyftError):
            return remote_res

        challenge_signature, remote_node_peer = remote_res

        # Verifying if the challenge is valid
        remote_node_verify_key.verify_key.verify(random_challenge, challenge_signature)

        # save the remote peer for later
        result = self.stash.update_peer(context.node.verify_key, remote_node_peer)
        if result.is_err():
            return SyftError(message=str(result.err()))

        return SyftSuccess(message="Routes Exchanged")

    @service_method(path="network.add_peer", name="add_peer", roles=GUEST_ROLE_LEVEL)
    def add_peer(
        self,
        context: AuthedServiceContext,
        peer: NodePeer,
        challenge: bytes,
        self_node_route: NodeRoute,
        verify_key: SyftVerifyKey,
    ) -> Union[bytes, SyftError]:
        """Add a Network Node Peer"""

        # Using the verify_key of the peer to verify the signature
        # It is also our single source of truth for the peer
        if peer.verify_key != context.credentials:
            return SyftError(
                message=(
                    f"The {type(peer)}.verify_key: "
                    f"{peer.verify_key} does not match the signature of the message"
                )
            )

        if verify_key != context.node.verify_key:
            return SyftError(
                message="verify_key does not match the remote node's verify_key for add_peer"
            )

        result = self.stash.update_peer(context.node.verify_key, peer)
        if result.is_err():
            return SyftError(message=str(result.err()))

        # this way they can match up who we are with who they think we are
        # Sending a signed messages for the peer to verify
        self_node_peer = self_node_route.validate_with_context(context=context)

        if isinstance(self_node_peer, SyftError):
            return self_node_peer

        # Q,TODO: Should the returned node peer also be signed
        # as the challenge is already signed

        challenge_signature = context.node.signing_key.signing_key.sign(
            challenge
        ).signature

        return [challenge_signature, self_node_peer]

    @service_method(path="network.ping", name="ping")
    def ping(
        self, context: AuthedServiceContext, challenge: bytes
    ) -> Union[bytes, SyftError]:
        """To check alivesness/authenticity of a peer"""

        # Only the root user can ping the node to check its state
        if context.node.verify_key != context.credentials:
            return SyftError(message=("Only the root user can access ping endpoint"))

        # this way they can match up who we are with who they think we are
        # Sending a signed messages for the peer to verify
        challenge_signature = context.node.signing_key.signing_key.sign(
            challenge
        ).signature

        return challenge_signature

    @service_method(path="network.add_route_for", name="add_route_for")
    def add_route_for(
        self,
        context: AuthedServiceContext,
        route: NodeRoute,
        peer: NodePeer,
    ) -> Union[SyftSuccess, SyftError]:
        """Add Route for this Node to another Node"""
        # check root user is asking for the exchange
        client = peer.client_with_context(context=context)
        result = client.api.services.network.verify_route(route)

        if not isinstance(result, SyftSuccess):
            return result
        return SyftSuccess(message="Route Verified")

    @service_method(
        path="network.verify_route", name="verify_route", roles=GUEST_ROLE_LEVEL
    )
    def verify_route(
        self, context: AuthedServiceContext, route: NodeRoute
    ) -> Union[SyftSuccess, SyftError]:
        """Add a Network Node Route"""
        # get the peer asking for route verification from its verify_key
        peer = self.stash.get_for_verify_key(
            context.node.verify_key, context.credentials
        )
        if peer.is_err():
            return SyftError(message=peer.err())
        peer = peer.ok()

        if peer.verify_key != context.credentials:
            return SyftError(
                message=(
                    f"verify_key: {context.credentials} at route {route} "
                    f"does not match listed peer: {peer}"
                )
            )
        peer.update_routes([route])
        result = self.stash.update_peer(context.node.verify_key, peer)
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="Network Route Verified")

    @service_method(
        path="network.get_all_peers", name="get_all_peers", roles=GUEST_ROLE_LEVEL
    )
    def get_all_peers(
        self, context: AuthedServiceContext
    ) -> Union[List[NodePeer], SyftError]:
        """Get all Peers"""
        result = self.stash.get_all(credentials=context.node.verify_key)
        if result.is_ok():
            peers = result.ok()
            return peers
        return SyftError(message=result.err())

    @service_method(
        path="network.get_peer_by_name", name="get_peer_by_name", roles=GUEST_ROLE_LEVEL
    )
    def get_peer_by_name(
        self, context: AuthedServiceContext, name: str
    ) -> Union[Optional[NodePeer], SyftError]:
        """Get Peer by Name"""
        result = self.stash.get_by_name(
            credentials=context.node.verify_key,
            name=name,
        )
        if result.is_ok():
            peer = result.ok()
            return peer
        return SyftError(message=str(result.err()))

    @service_method(
        path="network.get_peers_by_type",
        name="get_peers_by_type",
        roles=GUEST_ROLE_LEVEL,
    )
    def get_peers_by_type(
        self, context: AuthedServiceContext, node_type: NodeType
    ) -> Union[List[NodePeer], SyftError]:
        result = self.stash.get_by_node_type(
            credentials=context.node.verify_key, node_type=node_type
        )

        if result.is_err():
            return SyftError(message=str(result.err()))

        # Return peers or an empty list when result is None
        return result.ok() or []

    @service_method(path="network.join_vpn", name="join_vpn")
    def join_vpn(
        self,
        context: AuthedServiceContext,
        peer: Optional[NodePeer] = None,
        client: Optional[SyftClient] = None,
    ) -> Union[SyftSuccess, SyftError]:
        """Join a VPN Service"""

        if isinstance(client, SyftClient):
            remote_peer = NodePeer.from_client(client)
        else:
            remote_peer = peer
        if remote_peer is None:
            return SyftError("join_vpn requires peer or client")

        result = self.stash.get_by_uid(
            credentials=context.node.verify_key, uid=remote_peer.id
        )

        if result.is_err():
            return SyftError(message=f"{result.err()}")

        if result.ok() is not None:
            return SyftError(
                message=f"Already connected to VPN Peer: {remote_peer.name}"
            )

        # tell the remote peer our details
        if not context.node:
            return SyftError(message=f"{type(context)} has no node")

        # switch to the nodes signing key
        client = remote_peer.client_with_context(context=context)

        auth_token = client.api.services.network.register_to_vpn()

        if isinstance(auth_token, SyftError):
            return auth_token

        result = get_vpn_client(TailscaleClient)

        if result.is_err():
            return SyftError(message=result.err())

        tailscale_client = result.ok()

        result = tailscale_client.disconnect()

        if isinstance(result, SyftError):
            return result

        # TODO: move this url information /vpn stuff to the client
        vpn_url = GridURL.from_url(client.connection.url).with_path(path="/vpn")

        result = tailscale_client.connect(
            headscale_host=vpn_url,
            headscale_auth_token=auth_token.key,
        )

        if isinstance(result, SyftError):
            return result

        # save vpn token information to peer
        remote_peer.vpn_auth_key = auth_token.key
        remote_peer.is_vpn = True

        # save the remote peer for later
        result = self.stash.update_peer(
            credentials=context.node.verify_key, peer=remote_peer
        )
        if result.is_err():
            return SyftError(message=str(result.err()))

        if result.is_err():
            return SyftError(message=str(result.err()))

        return SyftSuccess(
            message=f"Successfully joined {remote_peer.name} via VPN !!!"
        )

    @service_method(path="network.vpn_status", name="vpn_status")
    def get_vpn_status(
        self,
        context: AuthedServiceContext,
    ) -> Union[TailscaleStatus, SyftError]:
        """Join a VPN Service"""
        result = get_vpn_client(TailscaleClient)

        if result.is_err():
            return SyftError(message=result.err())

        tailscale_client = result.ok()

        return tailscale_client.status()

    @service_method(
        path="network.register_to_vpn",
        name="register_to_vpn",
        roles=GUEST_ROLE_LEVEL,
    )
    def register_to_vpn(
        self,
        context: AuthedServiceContext,
    ) -> Union[HeadscaleAuthToken, SyftError]:
        """Register node to the VPN."""

        result = get_vpn_client(HeadscaleClient)

        if result.is_err():
            return SyftError(message=result.err())

        headscale_client = result.ok()

        token = headscale_client.generate_token()

        return token

    def connect_self(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        tailscale_status = self.get_vpn_status(context=context)

        if isinstance(tailscale_status, SyftError):
            return tailscale_status

        if tailscale_status.state is TailscaleState.RUNNING.value:
            return SyftSuccess(message="Connection already established !!")

        auth_token = self.register_to_vpn(context=context)

        if isinstance(auth_token, SyftError):
            return auth_token

        result = get_vpn_client(TailscaleClient)

        if result.is_err():
            return SyftError(message=result.err())

        tailscale_client = result.ok()

        result = tailscale_client.disconnect()

        if isinstance(result, SyftError):
            return result

        result = tailscale_client.connect(
            headscale_host="http://headscale:8080",
            headscale_auth_token=auth_token.key,
        )

        if isinstance(result, SyftError):
            return result

        return SyftSuccess(message="Successfully joined VPN !!!")


TYPE_TO_SERVICE[NodePeer] = NetworkService
SERVICE_TO_TYPES[NetworkService].update({NodePeer})


def from_grid_url(context: TransformContext) -> TransformContext:
    url = context.obj.url.as_container_host()
    context.output["host_or_ip"] = url.host_or_ip
    context.output["protocol"] = url.protocol
    context.output["port"] = url.port
    context.output["private"] = False
    context.output["proxy_target_uid"] = context.obj.proxy_target_uid
    return context


@transform(HTTPConnection, HTTPNodeRoute)
def http_connection_to_node_route() -> List[Callable]:
    return [from_grid_url]


def get_python_node_route(context: TransformContext) -> TransformContext:
    context.output["id"] = context.obj.node.id
    context.output["worker_settings"] = WorkerSettings.from_node(context.obj.node)
    context.output["proxy_target_uid"] = context.obj.proxy_target_uid
    return context


@transform(PythonConnection, PythonNodeRoute)
def python_connection_to_node_route() -> List[Callable]:
    return [get_python_node_route]


@transform_method(PythonNodeRoute, PythonConnection)
def node_route_to_python_connection(
    obj: Any, context: Optional[TransformContext] = None
) -> List[Callable]:
    return PythonConnection(node=obj.node, proxy_target_uid=obj.proxy_target_uid)


@transform_method(HTTPNodeRoute, HTTPConnection)
def node_route_to_http_connection(
    obj: Any, context: Optional[TransformContext] = None
) -> List[Callable]:
    url = GridURL(
        protocol=obj.protocol, host_or_ip=obj.host_or_ip, port=obj.port
    ).as_container_host()
    return HTTPConnection(url=url, proxy_target_uid=obj.proxy_target_uid)


@transform(NodeMetadata, NodePeer)
def metadata_to_peer() -> List[Callable]:
    return [
        keep(["id", "name", "verify_key", "node_type"]),
    ]
