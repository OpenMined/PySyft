from typing import Callable

# external class imports
from nacl.signing import VerifyKey

# syft imports
from ....common.message import SyftMessage
from ...abstract.node import AbstractNode


class AuthorizationException(Exception):
    pass


def service_auth(
    root_only=False,
    existing_users_only=False,
    guests_welcome=False,
    register_new_guests=False,
):
    def decorator(func: Callable) -> Callable:
        def process(
            node: AbstractNode, msg: SyftMessage, verify_key: VerifyKey
        ) -> SyftMessage:
            print(f"> Checking {msg.pprint} 🔑 Matches {node.pprint} root 🗝")
            if root_only:
                if verify_key != node.root_verify_key:
                    print(f"> ❌ Auth FAILED {msg.pprint} 🔑 != 🗝")
                    raise AuthorizationException(
                        "You are not Authorized to access this service"
                    )
                else:
                    print(f"> ✅ Auth Succeeded {msg.pprint} 🔑 == 🗝")

            elif existing_users_only:
                assert verify_key in node.guest_verify_key_registry
            elif guests_welcome:

                if register_new_guests:
                    node.guest_verify_key_registry.add(verify_key)

            else:
                raise Exception("You must configure services auth with a flag.")

            return func(node=node, msg=msg, verify_key=verify_key)

        return process

    return decorator
