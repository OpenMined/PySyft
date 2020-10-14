# stdlib
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.signing import SigningKey

# syft relative
from ..core.io.connection import ClientConnection
from ..core.io.location.specific import SpecificLocation
from ..core.io.route import SoloRoute
from ..core.node.common.client import Client
from ..core.node.device.client import DeviceClient
from ..core.node.domain.client import DomainClient
from ..core.node.network.client import NetworkClient
from ..core.node.vm.client import VirtualMachineClient
from ..decorators.syft_decorator_impl import syft_decorator


def connect(
    credentials: Dict,
    url: str,
    conn_type: ClientConnection,
    client_type: Client,
) -> Any:
    class GridClient(client_type):  # type: ignore
        def __init__(
            self,
            credentials: Dict,
            url: str,
            conn_type: ClientConnection,
            client_type: Client,
        ) -> None:
            # Load an Signing Key instance
            signing_key = SigningKey.generate()
            verify_key = signing_key.verify_key

            # Use Signaling Server metadata
            # to build client route
            conn = conn_type(url=url)  # type: ignore
            metadata, user_key = conn.login(credentials=credentials)

            (
                spec_location,
                name,
                client_id,
            ) = client_type.deserialize_client_metadata_from_node(metadata=metadata)

            # Create a new Solo Route using the selected connection type
            route = SoloRoute(destination=spec_location, connection=conn)

            location_args = self.__route_client_location(
                client_type=client_type, location=spec_location
            )

            # Create a new signaling client using the selected client type
            super().__init__(
                network=location_args[NetworkClient],
                domain=location_args[DomainClient],
                device=location_args[DeviceClient],
                vm=location_args[VirtualMachineClient],
                name=name,
                routes=[route],
                signing_key=signing_key,
                verify_key=verify_key,
            )

        @syft_decorator(typechecking=True)
        def __route_client_location(
            self, client_type: Any, location: SpecificLocation
        ) -> Dict:
            locations: Dict[Any, Optional[SpecificLocation]] = {
                NetworkClient: None,
                DomainClient: None,
                DeviceClient: None,
                VirtualMachineClient: None,
            }
            locations[client_type] = location
            return locations

    return GridClient(
        credentials=credentials, url=url, conn_type=conn_type, client_type=client_type
    )
