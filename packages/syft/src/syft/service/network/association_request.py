# stdlib
import secrets
from typing import cast

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...abstract_node import AbstractNode
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ..context import ChangeContext
from ..request.request import Change
from ..response import SyftError
from ..response import SyftSuccess
from .routes import NodeRoute


@serializable()
class AssociationRequestChange(Change):
    __canonical_name__ = "AssociationRequestChange"
    __version__ = SYFT_OBJECT_VERSION_1

    self_node_route: NodeRoute
    remote_node_route: NodeRoute
    remote_node_verify_key: SyftVerifyKey

    __repr_attrs__ = ["self_node_route", "remote_node_route", "remote_node_verify_key"]

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[SyftSuccess, SyftError]:
        # relative
        from .network_service import NetworkService

        if not apply:
            return SyftError(message="Undo not supported for AssociationRequestChange")

        service_ctx = context.to_service_ctx()

        # Step 1: Validate the Route
        self_node_peer = self.self_node_route.validate_with_context(context=service_ctx)

        if isinstance(self_node_peer, SyftError):
            return self_node_peer

        # Step 2: Send the Node Peer to the remote node
        # Also give them their own to validate that it belongs to them
        # random challenge prevents replay attacks
        remote_client = self.remote_node_route.client_with_context(context=service_ctx)
        random_challenge = secrets.token_bytes(16)

        remote_res = remote_client.api.services.network.add_peer(
            peer=self_node_peer,
            challenge=random_challenge,
            self_node_route=self.remote_node_route,
            verify_key=self.remote_node_verify_key,
        )

        if isinstance(remote_res, SyftError):
            return Err(remote_res)

        challenge_signature, remote_node_peer = remote_res

        # Verifying if the challenge is valid

        try:
            self.remote_node_verify_key.verify_key.verify(
                random_challenge, challenge_signature
            )
        except Exception as e:
            return Err(SyftError(message=str(e)))

        # save the remote peer for later
        context.node = cast(AbstractNode, context.node)

        network_stash = context.node.get_service(NetworkService).stash

        result = network_stash.update_peer(context.node.verify_key, remote_node_peer)
        if result.is_err():
            return Err(SyftError(message=str(result.err())))

        return Ok(SyftSuccess(message="Routes Exchanged"))

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context, apply=False)

    def __repr_syft_nested__(self) -> str:
        return (
            f"Request for connecting {self.self_node_route} -> {self.remote_node_route}"
        )
