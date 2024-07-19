# stdlib
import secrets
from typing import cast

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...client.client import SyftClient
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ..context import ChangeContext
from ..request.request import Change
from ..response import SyftError
from ..response import SyftSuccess
from .routes import ServerRoute
from .server_peer import ServerPeer


@serializable()
class AssociationRequestChange(Change):
    __canonical_name__ = "AssociationRequestChange"
    __version__ = SYFT_OBJECT_VERSION_1

    self_server_route: ServerRoute
    remote_peer: ServerPeer
    challenge: bytes

    __repr_attrs__ = ["self_server_route", "remote_peer"]

    def _run(
        self, context: ChangeContext, apply: bool
    ) -> Result[tuple[bytes, ServerPeer], SyftError]:
        """
        Executes the association request.

        Args:
            context (ChangeContext): The change context.
            apply (bool): A flag indicating whether to apply the association request.

        Returns:
            Result[tuple[bytes, ServerPeer], SyftError]: The result of the association request.
        """
        # relative
        from .network_service import NetworkService

        if not apply:
            # TODO: implement undo for AssociationRequestChange
            return Err(
                SyftError(message="Undo not supported for AssociationRequestChange")
            )

        # Get the network service
        service_ctx = context.to_service_ctx()
        network_service = cast(
            NetworkService, service_ctx.server.get_service(NetworkService)
        )
        network_stash = network_service.stash

        # Check if remote peer to be added is via reverse tunnel
        rtunnel_route = self.remote_peer.get_rtunnel_route()
        add_rtunnel_route = (
            rtunnel_route is not None
            and self.remote_peer.latest_added_route == rtunnel_route
        )

        # If the remote peer is added via reverse tunnel, we skip ping to peer
        if add_rtunnel_route:
            network_service.set_reverse_tunnel_config(
                context=context,
                remote_server_peer=self.remote_peer,
            )
        else:
            # Pinging the remote peer to verify the connection
            try:
                remote_client: SyftClient = self.remote_peer.client_with_context(
                    context=service_ctx
                )
                if remote_client.is_err():
                    return SyftError(
                        message=f"Failed to create remote client for peer: "
                        f"{self.remote_peer.id}. Error: {remote_client.err()}"
                    )
                remote_client = remote_client.ok()
                random_challenge = secrets.token_bytes(16)
                remote_res = remote_client.api.services.network.ping(
                    challenge=random_challenge
                )
            except Exception as e:
                return SyftError(message="Remote Peer cannot ping peer:" + str(e))

            if isinstance(remote_res, SyftError):
                return Err(remote_res)

            challenge_signature = remote_res

            # Verifying if the challenge is valid
            try:
                self.remote_peer.verify_key.verify_key.verify(
                    random_challenge, challenge_signature
                )
            except Exception as e:
                return Err(SyftError(message=str(e)))

        # Adding the remote peer to the network stash
        result = network_stash.create_or_update_peer(
            service_ctx.server.verify_key, self.remote_peer
        )

        if result.is_err():
            return Err(SyftError(message=str(result.err())))

        # this way they can match up who we are with who they think we are
        # Sending a signed messages for the peer to verify
        self_server_peer = self.self_server_route.validate_with_context(
            context=service_ctx
        )

        if isinstance(self_server_peer, SyftError):
            return Err(self_server_peer)

        return Ok(
            SyftSuccess(
                message=f"Routes successfully added for peer: {self.remote_peer.name}"
            )
        )

    def apply(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context, apply=True)

    def undo(self, context: ChangeContext) -> Result[SyftSuccess, SyftError]:
        return self._run(context, apply=False)

    def __repr_syft_nested__(self) -> str:
        return f"Request for connection from : {self.remote_peer.name}"
