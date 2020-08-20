# external class imports
from typing import Callable
from nacl.signing import VerifyKey

# syft imports
from ....common.message import SyftMessage
from ...abstract.node import AbstractNode


def service_auth(
    root_only: bool = False,
    existing_users_only: bool = False,
    guests_welcome: bool = False,
    register_new_guests: bool = False,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        def process(
            node: AbstractNode, msg: SyftMessage, verify_key: VerifyKey
        ) -> SyftMessage:

            if root_only:
                if verify_key != node.root_verify_key:
                    raise Exception("User is not root.")

            elif existing_users_only:
                if verify_key not in node.guest_verify_key_registry:
                    raise Exception("User not known.")

            elif guests_welcome:
                if register_new_guests:
                    node.guest_verify_key_registry.add(verify_key)

            else:
                raise Exception("You must configure services auth with a flag.")

            return func(node=node, msg=msg, verify_key=verify_key)

        return process

    return decorator
