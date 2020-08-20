import pytest
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

import syft as sy

from syft.core.node.common.action.auth import service_auth


def test_service_auth():
    node = sy.Device()
    msg = sy.ReprMessage(address=node.address)

    random_signing_key = SigningKey.generate()
    random_verify_key = random_signing_key.verify_key

    # Root only
    @service_auth(root_only=True)
    def process(node: sy.Device, msg: sy.ReprMessage, verify_key: VerifyKey) -> None:
        print(node)

    process(node=node, msg=msg, verify_key=node.root_verify_key)

    with pytest.raises(Exception, match="User is not root."):
        process(node=node, msg=msg, verify_key=random_verify_key)
