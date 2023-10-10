# relative
from ...serde.deserialize import _deserialize
from ...serde.serializable import recursive_serde_register
from ...serde.serialize import _serialize
from .auth import login  # noqa: F401
from .deployment import create_deployment  # noqa: F401
from .oblv_proxy import check_oblv_proxy_installation_status  # noqa: F401
from .oblv_proxy import create_oblv_key_pair  # noqa: F401
from .oblv_proxy import get_oblv_public_key  # noqa: F401
from .oblv_proxy import install_oblv_proxy  # noqa: F401

try:
    # third party
    from oblv_ctl.oblv_client import OblvClient

    # Oblivious Client serde
    recursive_serde_register(
        OblvClient,
        serialize=lambda x: _serialize([x.token, x.oblivious_user_id], to_bytes=True),
        deserialize=lambda x: OblvClient(*_deserialize(x, from_bytes=True)),
    )

except Exception:  # nosec
    pass
