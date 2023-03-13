# third party
import requests
from result import Err
from result import Ok
from result import Result

# relative
from .context import AuthedServiceContext
from .serializable import serializable
from .service import AbstractService
from .service import service_method


@serializable(recursive_serde=True)
class TestService(AbstractService):
    def __init__(self) -> None:
        pass

    @service_method(path="test.send", name="send_name")
    def send_name(self, context: AuthedServiceContext, name: str) -> Result[Ok, Err]:
        """Initial testing service"""

        result = f"Hello {name}"
        return Ok(result)

    @service_method(path="test.request", name="request")
    def test_request(self, context: AuthedServiceContext, url: str):
        res = requests.get(url)  # nosec
        return Ok(res.status_code)
