# relative
from ..core.node.new.deserialize import _deserialize
from ..core.node.new.recursive import recursive_serde_register
from ..core.node.new.serialize import _serialize

INFRA = "m5.4xlarge"
REPO_OWNER = "OpenMined"
REPO_NAME = "syft-enclave"
REF = "manual_code"
REGION = "us-west-2"
VCS = "github"
VISIBILITY = "private"
LOCAL_MODE = True
DOMAIN_CONNECTION_PORT = 3030
WORKER_MODE = False  # Used for testing with the enclave being the in-memory worker


try:
    # third party
    from oblv.oblv_client import OblvClient

    # Oblivious Client serde
    recursive_serde_register(
        OblvClient,
        serialize=lambda x: _serialize([x.token, x.oblivious_user_id], to_bytes=True),
        deserialize=lambda x: OblvClient(*_deserialize(x, from_bytes=True)),
    )

except Exception:  # nosec
    pass
