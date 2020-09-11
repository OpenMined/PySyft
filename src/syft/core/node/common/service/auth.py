# stdlib
from typing import Callable
from typing import Optional

# third party
from nacl.signing import VerifyKey

# syft absolute
import syft as sy

# syft relative
from ....common.message import SyftMessage
from ...abstract.node import AbstractNode


class AuthorizationException(Exception):
    pass


def service_auth(
    root_only: bool = False,
    existing_users_only: bool = False,
    guests_welcome: bool = False,
    register_new_guests: bool = False,
) -> Callable:
    def decorator(func: Callable) -> Callable:
        def process(
            node: AbstractNode, msg: SyftMessage, verify_key: VerifyKey
        ) -> Optional[SyftMessage]:
            if sy.VERBOSE:
                print(f"> Checking {msg.pprint} 🔑 Matches {node.pprint} root 🗝")

            if root_only:
                if sy.VERBOSE:
                    keys = (
                        f"> Matching 🔑 {node.key_emoji(key=verify_key)}  == "
                        + f"{node.key_emoji(key=node.root_verify_key)}  🗝"
                    )
                    print(keys)
                if verify_key != node.root_verify_key:
                    if sy.VERBOSE:
                        print(f"> ❌ Auth FAILED {msg.pprint}")
                    raise AuthorizationException(
                        "You are not Authorized to access this service"
                    )
                else:
                    if sy.VERBOSE:
                        print(f"> ✅ Auth Succeeded {msg.pprint} 🔑 == 🗝")

            elif existing_users_only:
                if verify_key not in node.guest_verify_key_registry:
                    raise AuthorizationException("User not known.")

            elif guests_welcome:
                if register_new_guests:
                    node.guest_verify_key_registry.add(verify_key)

            else:
                raise Exception("You must configure services auth with a flag.")

            # Can be None because not all functions reply
            return func(node=node, msg=msg, verify_key=verify_key)

        return process

    return decorator
