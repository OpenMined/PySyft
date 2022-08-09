# stdlib
from typing import Any

# syft absolute
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNJoinSelfMessageWithReply,
)
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNStatusMessageWithReply,
)
from syft.core.node.common.node_service.vpn.vpn_messages import TAILSCALE_URL
from syft.core.node.common.node_service.vpn.vpn_messages import connect_with_key
from syft.core.node.common.node_service.vpn.vpn_messages import get_network_url

# grid absolute
from grid.core.celery_app import celery_app
from grid.core.config import settings  # noqa: F401
from grid.core.node import node
from grid.periodic_tasks import cleanup_incomplete_uploads_from_blob_store

# TODO : Should be modified to use exponential backoff (for efficiency)
# Initially we have set 0.1 as the retry time.
# We have set max retries =(1200) 120 seconds


@celery_app.task(bind=True, acks_late=True)
def msg_without_reply(self, obj_msg: Any) -> None:  # type: ignore
    if isinstance(obj_msg, SignedImmediateSyftMessageWithoutReply):
        try:
            node.recv_immediate_msg_without_reply(msg=obj_msg)
        except Exception as e:
            raise e
    else:
        raise Exception(
            f"This worker can only handle SignedImmediateSyftMessageWithoutReply. {obj_msg}"
        )


@celery_app.task
def network_connect_self_task() -> None:
    network_connect_self()


@celery_app.task
def domain_reconnect_network_task() -> None:
    domain_reconnect_network()


def domain_reconnect_network() -> None:
    for node_connections in node.node.all():
        network_vpn_endpoint = get_network_url().with_path("/api/v1/vpn/status")
        msg = (
            VPNStatusMessageWithReply(kwargs={})
            .to(address=node.address, reply_to=node.address)
            .sign(signing_key=node.signing_key)
        )
        reply = node.recv_immediate_msg_with_reply(msg=msg)
        is_connected = reply.message.payload.kwargs["connected"]
        disconnected = not is_connected or not network_vpn_endpoint
        if node_connections.keep_connected and disconnected:
            routes = list(node.node_route.query(node_id=node_connections.id))
            for route in routes:
                try:
                    status, error = connect_with_key(
                        tailscale_host=TAILSCALE_URL,
                        headscale_host=route.vpn_endpoint,
                        vpn_auth_key=route.vpn_key,
                    )

                    if not status:
                        print("connect with key failed", error)
                # If for some reason this route didn't work (ex: network node offline, etc),
                # just skip it, show the error and go to the next one.
                except Exception as e:
                    print("error: ", str(e))
                    continue


def network_connect_self() -> None:
    # TODO: refactor to be non blocking and in a different queue
    msg = (
        VPNJoinSelfMessageWithReply(kwargs={})
        .to(address=node.address, reply_to=node.address)
        .sign(signing_key=node.signing_key)
    )
    _ = node.recv_immediate_msg_with_reply(msg=msg).message


@celery_app.on_after_configure.connect
def add_cleanup_blob_store_periodic_task(sender, **kwargs) -> None:  # type: ignore
    celery_app.add_periodic_task(
        3600,  # Run every hour
        cleanup_incomplete_uploads_from_blob_store.s(),
        name="Clean incomplete uploads in Seaweed",
        queue="main-queue",
        options={"queue": "main-queue"},
    )


if settings.NODE_TYPE.lower() == "network":
    network_connect_self()

    @celery_app.on_after_configure.connect
    def add_network_connect_self_periodic_task(sender, **kwargs) -> None:  # type: ignore
        celery_app.add_periodic_task(
            settings.NETWORK_CHECK_INTERVAL,  # Run every second
            network_connect_self_task.s(),
            name="Connect Network VPN to itself",
            queue="main-queue",
            options={"queue": "main-queue"},
        )


if settings.NODE_TYPE.lower() == "domain":

    @celery_app.on_after_configure.connect
    def add_domain_reconnect_periodic_task(sender, **kwargs) -> None:  # type: ignore
        celery_app.add_periodic_task(
            settings.DOMAIN_CHECK_INTERVAL,  # Run every second
            domain_reconnect_network_task.s(),
            name="Reconnect Domain to Network VPN",
            queue="main-queue",
            options={"queue": "main-queue"},
        )
