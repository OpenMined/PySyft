# stdlib
from io import StringIO
import sys
from typing import Any
from typing import Dict

# third party
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import safe_builtins
from celery.utils.log import get_task_logger

# syft absolute
from syft.core.common.group import VERIFYALL
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.common.uid import UID
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNJoinSelfMessageWithReply,
)
from syft.core.node.common.node_service.vpn.vpn_messages import (
    VPNStatusMessageWithReply,
)
from syft.core.node.common.node_service.vpn.vpn_messages import TAILSCALE_URL
from syft.core.node.common.node_service.vpn.vpn_messages import connect_with_key
from syft.core.node.common.node_service.vpn.vpn_messages import get_network_url
from syft.core.store.storeable_object import StorableObject

# grid absolute
from grid.core.celery_app import celery_app
from grid.core.config import settings  # noqa: F401
from grid.core.node import node
from grid.periodic_tasks import check_tasks_to_be_executed
from grid.periodic_tasks import cleanup_incomplete_uploads_from_blob_store

logger = get_task_logger(__name__)


# TODO : Should be modified to use exponential backoff (for efficiency)
# Initially we have set 0.1 as the retry time.
# We have set max retries=(1200) 120 seconds
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


stdout_ = sys.stdout
stderr_ = sys.stderr


restricted_globals = dict(__builtins__=safe_builtins)


@celery_app.task
def execute_task(
    task_uid: str, code: str, inputs: Dict[str, str], outputs: Dict[str, str]
) -> None:
    global stdout_
    global stderr_
    try:
        node.tasks.update(
            search_params={"uid": task_uid},
            updated_args={"execution": {"status": "executing"}},
        )

        # Check overlap between inputs and vars
        global_input_inter = set(globals().keys()).intersection(set(inputs.keys()))
        local_input_inter = set(vars().keys()).intersection(set(inputs.keys()))

        # If there's some intersection between global variables and input
        if global_input_inter or local_input_inter:
            stderr_message = " You can't use variable name: "
            stderr_message += ",".join(list(global_input_inter))
            stderr_message += ",".join(list(local_input_inter))

            node.tasks.update(
                search_params={"uid": task_uid},
                updated_args={
                    "execution": {"status": "failed", "stderr": stderr_message}
                },
            )
            return None

        local_vars = {}
        for key, value in inputs.items():
            local_vars[key] = node.store.get(value, proxy_only=True).data

        # create file-like string to capture ouputs
        codeOut = StringIO()
        codeErr = StringIO()

        sys.stdout = codeOut
        sys.stderr = codeErr

        old_env = set(vars().keys())

        byte_code = compile_restricted(code, "<string>", "exec")
        exec(byte_code, restricted_globals)

        method_name = list(
            set([key for key in vars().keys() if key != "old_env"]) - old_env
        )[0]
        outputs = vars()[method_name](**local_vars)

        # restore stdout and stderr
        sys.stdout = stdout_
        sys.stderr = stderr_

        logger.info(outputs)

        logger.info("Error: " + str(codeErr.getvalue()))
        logger.info("Std ouputs: " + str(codeOut.getvalue()))

        # _save_vars = {}
        # for var in outputs.keys():
        new_id = UID()
        node.store.check_collision(new_id)

        obj = StorableObject(
            id=new_id,
            data=outputs,
            search_permissions={VERIFYALL: None},
        )

        obj.read_permissions = {
            node.verify_key: node.id,
        }

        obj.write_permissions = {
            node.verify_key: node.id,
        }

        node.store[new_id] = obj

        node.tasks.update(
            search_params={"uid": task_uid},
            updated_args={
                "execution": {"status": "done"},
                "outputs": {"output": new_id.to_string()},
            },
        )
    except Exception as e:
        sys.stdout = stdout_
        sys.stderr = stderr_
        print("Task Failed with Exception", e)
    finally:
        sys.stdout = stdout_
        sys.stderr = stderr_


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
            .to(address=node.node_uid, reply_to=node.node_uid)
            .sign(signing_key=node.signing_key)
        )
        reply = node.recv_immediate_msg_with_reply(msg=msg)
        is_connected = reply.message.payload.kwargs["connected"]
        disconnected = not is_connected or not network_vpn_endpoint
        if node_connections.keep_connected and disconnected:
            routes = node.node.get_routes(node_connections)

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
        .to(address=node.node_uid, reply_to=node.node_uid)
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


@celery_app.on_after_configure.connect
def check_ready_tasks(sender, **kwargs) -> None:  # type: ignore
    celery_app.add_periodic_task(
        3,  # Run every hour
        check_tasks_to_be_executed.s(),
        name="Check tasks that are ready to be executed ...",
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
