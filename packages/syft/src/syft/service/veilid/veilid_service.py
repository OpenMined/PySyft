# stdlib
from typing import Union

# third party
import requests

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...util.telemetry import instrument
from ..context import AuthedServiceContext
from ..network.routes import VeilidNodeRoute
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from .veilid_endpoints import GEN_DHT_KEY_ENDPOINT
from .veilid_endpoints import HEALTHCHECK_ENDPOINT
from .veilid_endpoints import RET_DHT_KEY_ENDPOINT
from .veilid_endpoints import VEILID_SERVICE_URL


@instrument
@serializable()
class VeilidService(AbstractService):
    store: DocumentStore

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    @service_method(
        path="veilid.generate_dht_key",
        name="generate_dht_key",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def generate_dht_key(
        self, context: AuthedServiceContext
    ) -> Union[SyftSuccess, SyftError]:
        # TODO: Simplify the below logic related to HARDCODED Strings
        status_res = self.check_veilid_status()
        if isinstance(status_res, SyftError):
            return status_res
        try:
            response = requests.post(
                f"{VEILID_SERVICE_URL}{GEN_DHT_KEY_ENDPOINT}",
            )
            if (
                response.status_code == 200
                and response.json().get("message") == "DHT Key generated successfully"
            ):
                return SyftSuccess(message="DHT key generated successfully")

            return SyftError(message=f"Failed to generate DHT key. {response.json()}")
        except Exception as e:
            return SyftError(message=f"Failed to generate DHT key. {e}")

    @service_method(
        path="veilid.retrieve_dht_key",
        name="retrieve_dht_key",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def retrieve_dht_key(self, context: AuthedServiceContext) -> Union[str, SyftError]:
        # TODO: Simplify the below logic related to HARDCODED Strings
        status_res = self.check_veilid_status()
        if isinstance(status_res, SyftError):
            return status_res
        try:
            response = requests.get(
                f"{VEILID_SERVICE_URL}{RET_DHT_KEY_ENDPOINT}",
            )
            if response.status_code == 200:
                if response.json().get("message") == "DHT Key does not exist":
                    return SyftError(
                        message="DHT key does not exist.Invoke .generate_dht_key to generate a new key."
                    )
                else:
                    return response.json().get("message")
            return SyftError(
                message=f"Failed to retrieve DHT key. status_code:{response.status_code} error: {response.json()}"
            )
        except Exception as e:
            return SyftError(message=f"Failed to retrieve DHT key. exception: {e}")

    @staticmethod
    def check_veilid_status() -> Union[SyftSuccess, SyftError]:
        status = False
        try:
            response = requests.get(f"{VEILID_SERVICE_URL}{HEALTHCHECK_ENDPOINT}")
            if response.status_code == 200 and response.json().get("message") == "OK":
                status = True
        except Exception:
            pass

        if status:
            return SyftSuccess(message="Veilid service is healthy.")
        else:
            return SyftError(
                message="Veilid service is not healthy. Please try again later."
            )

    @service_method(
        path="veilid.get_veilid_route",
        name="get_veilid_route",
    )
    def get_veilid_route(
        self, context: AuthedServiceContext
    ) -> Union[VeilidNodeRoute, SyftError]:
        dht_key = self.retrieve_dht_key(context)
        if isinstance(dht_key, SyftError):
            return dht_key
        return VeilidNodeRoute(dht_key=dht_key)
