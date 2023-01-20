# stdlib
import io
import json
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.encoding import HexEncoder
from nacl.signing import SigningKey
import requests
from requests.adapters import HTTPAdapter

# from requests.adapters import TimeoutHTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests_toolbelt.multipart.encoder import MultipartEncoder

# relative
from .. import GridURL
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.common.serde.serializable import serializable
from ...core.common.serde.serialize import _serialize
from ...core.node.common.exceptions import AuthorizationError
from ...core.node.enums import RequestAPIFields
from ...core.node.exceptions import RequestAPIException
from ...logger import debug
from ...proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB
from ...proto.grid.connections.http_connection_pb2 import (
    GridHTTPConnection as GridHTTPConnection_PB,
)
from ...util import verify_tls
from ..connections.http_connection import HTTPConnection

DEFAULT_TIMEOUT = 30  # seconds


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        if "max_retries" in kwargs:
            self.max_retries = kwargs["max_retries"]
            del kwargs["max_retries"]
        super().__init__(*args, **kwargs)

    def send(self, request: Any, **kwargs: Any) -> Any:  # type:ignore
        timeout = kwargs.get("timeout")
        if timeout is None:
            kwargs["timeout"] = self.timeout
        return super().send(request, **kwargs)


@serializable()
class GridHTTPConnection(HTTPConnection):

    LOGIN_ROUTE = "/login"
    KEY_ROUTE = "/key"
    GUEST_ROUTE = "/guest"
    SYFT_ROUTE = "/syft"
    SYFT_ROUTE_STREAM = "/syft/stream"  # non blocking node
    # SYFT_MULTIPART_ROUTE = "/pysyft_multipart"
    SIZE_THRESHOLD = 20971520  # 20 MB

    def __init__(self, url: Union[GridURL, str]) -> None:
        self.base_url = GridURL.from_url(url) if isinstance(url, str) else url
        if self.base_url is None:
            raise Exception(f"Invalid GridURL. {self.base_url}")
        self.session_token: str = ""
        self.token_type: str = "'"

    @property
    def header(self) -> Dict[str, str]:

        _header = {}

        if self.session_token and self.token_type:
            _header = dict(
                Authorization="Bearer "
                + json.loads(
                    '{"auth_token":"'
                    + str(self.session_token)
                    + '","token_type":"'
                    + str(self.token_type)
                    + '"}'
                )["auth_token"]
            )

        _header["Content-Type"] = "application/octet-stream"
        return _header

    def _send_msg(
        self, msg: SyftMessage, timeout: Optional[float] = None
    ) -> requests.Response:
        """
        Serializes Syft messages in json format and send it using HTTP protocol.
        NOTE: Auxiliary method to avoid code duplication and modularity.
        :return: returns requests.Response object containing a JSON serialized
        SyftMessage
        :rtype: requests.Response
        """

        header = self.header

        route = GridHTTPConnection.SYFT_ROUTE
        # if the message has no reply lets use the streaming endpoint
        # this allows the streaming endpoint to run on an entirely different process
        if isinstance(
            msg,
            (SignedImmediateSyftMessageWithoutReply, ImmediateSyftMessageWithoutReply),
        ):
            route = GridHTTPConnection.SYFT_ROUTE_STREAM

        # Perform HTTP request using base_url as a root address
        msg_bytes: bytes = _serialize(obj=msg, to_bytes=True)  # type: ignore

        # timeout = None will wait forever
        timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

        # if sys.getsizeof(msg_bytes) < GridHTTPConnection.SIZE_THRESHOLD:
        # if True:
        r = requests.post(
            url=str(self.base_url) + route,
            data=msg_bytes,
            headers=header,
            verify=verify_tls(),
            timeout=timeout,
            proxies=HTTPConnection.proxies,
        )
        if r.status_code == 401:
            raise AuthorizationError(
                "Check if your credentials are still valid or if your session was expired."
            )
        # else:
        #     r = self.send_streamed_messages(blob_message=msg_bytes)

        # Return request's response object
        # r.text provides the response body as a str
        return r

    def login(self, credentials: Dict) -> Tuple:
        if credentials:
            url = str(self.base_url) + GridHTTPConnection.LOGIN_ROUTE
        else:
            url = str(self.base_url) + GridHTTPConnection.GUEST_ROUTE

        response = requests.post(
            url=url,
            json=credentials,
            verify=verify_tls(),
            timeout=2,
            proxies=HTTPConnection.proxies,
        )

        # Response
        content = json.loads(response.text)
        # If fail
        if response.status_code != requests.codes.ok:
            raise Exception(content["detail"])

        metadata = content["metadata"].encode("ISO-8859-1")
        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(metadata)

        # If success
        # Save session token
        self.session_token = content["access_token"]
        self.token_type = content["token_type"]

        # Return node metadata / user private key
        return (metadata_pb, content["key"])

    def auth_using_key(self, user_key: SigningKey) -> Dict:
        response = requests.post(
            url=str(self.base_url) + GridHTTPConnection.KEY_ROUTE,
            json={"signing_key": user_key.encode(encoder=HexEncoder).decode("utf-8")},
            verify=verify_tls(),
            timeout=2,
            proxies=HTTPConnection.proxies,
        )
        # Response
        content = json.loads(response.text)
        # If fail
        if response.status_code != requests.codes.ok:
            raise Exception(content["detail"])

        metadata = content["metadata"].encode("ISO-8859-1")
        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(metadata)

        # If success
        # Save session token
        self.session_token = content["access_token"]
        self.token_type = content["token_type"]
        return metadata_pb

    def _get_metadata(self, timeout: Optional[float] = 2) -> Tuple:
        """Request Node's metadata
        :return: returns node metadata
        :rtype: str of bytes
        """
        # allow retry when connecting in CI
        session = requests.Session()
        retry = Retry(connect=1, backoff_factor=0.5)
        if timeout is None:
            adapter = HTTPAdapter(max_retries=retry)
        else:
            adapter = TimeoutHTTPAdapter(max_retries=retry, timeout=timeout)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        metadata_url = str(self.base_url) + "/syft/metadata"
        response = session.get(metadata_url, verify=verify_tls())

        if response.status_code != 200:
            raise requests.ConnectionError(
                f"Failed to fetch metadata. Response returned with code {response.status_code}"
            )

        # upgrade to tls if available
        try:
            if response.url.startswith("https://") and self.base_url.protocol == "http":
                # we got redirected to https
                self.base_url = GridURL.from_url(
                    response.url.replace("/syft/metadata", "")
                )
                debug(f"GridURL Upgraded to HTTPS. {self.base_url}")
        except Exception as e:
            print(f"Failed to upgrade to HTTPS. {e}")

        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(response.content)

        return metadata_pb

    def setup(self, **content: Dict[str, Any]) -> Any:
        response = json.loads(
            requests.post(
                str(self.base_url) + "/setup",
                json=content,
                verify=verify_tls(),
                timeout=2,
                proxies=HTTPConnection.proxies,
            ).text
        )
        if response.get(RequestAPIFields.MESSAGE, None):
            return response
        else:
            raise RequestAPIException(response.get(RequestAPIFields.ERROR))

    def reset(self) -> Any:
        header = {}

        if self.session_token and self.token_type:
            header = dict(
                Authorization="Bearer "
                + json.loads(
                    '{"auth_token":"'
                    + str(self.session_token)
                    + '","token_type":"'
                    + str(self.token_type)
                    + '"}'
                )["auth_token"]
            )

        response = json.loads(
            requests.delete(
                str(self.base_url) + GridHTTPConnection.SYFT_ROUTE,
                headers=header,
                verify=verify_tls(),
                proxies=HTTPConnection.proxies,
            ).text
        )
        if response.get(RequestAPIFields.MESSAGE, None, timeout=2):
            return response
        else:
            raise RequestAPIException(response.get(RequestAPIFields.ERROR))

    def send_files(
        self, route: str, file_path: str, form_name: str, form_values: Dict
    ) -> Dict[str, Any]:
        header = {}

        if self.session_token and self.token_type:
            header = dict(
                Authorization="Bearer "
                + json.loads(
                    '{"auth_token":"'
                    + str(self.session_token)
                    + '","token_type":"'
                    + str(self.token_type)
                    + '"}'
                )["auth_token"]
            )

        files = {
            form_name: (None, json.dumps(form_values), "text/plain"),
            "file": (file_path, open(file_path, "rb"), "application/octet-stream"),
        }

        resp = requests.post(
            str(self.base_url) + route,
            files=files,  # type: ignore
            headers=header,
            verify=verify_tls(),
            proxies=HTTPConnection.proxies,
        )

        return json.loads(resp.content)

    def send_streamed_messages(self, blob_message: bytes) -> requests.Response:
        session = requests.Session()
        with io.BytesIO(blob_message) as msg:
            form = MultipartEncoder(
                {
                    "file": ("message", msg.read(), "application/octet-stream"),
                }
            )

            headers = {
                "Prefer": "respond-async",
                "Content-Type": form.content_type,
            }

            resp = session.post(
                str(self.base_url) + GridHTTPConnection.SYFT_ROUTE_STREAM,
                headers=headers,
                data=form,
                verify=verify_tls(),
                proxies=HTTPConnection.proxies,
            )

        session.close()
        return resp

    @property
    def host(self) -> str:
        return self.base_url.base_url

    @staticmethod
    def _proto2object(proto: GridHTTPConnection_PB) -> "GridHTTPConnection":
        obj = GridHTTPConnection(url=GridURL.from_url(proto.base_url))
        obj.session_token = proto.session_token
        obj.token_type = proto.token_type
        return obj

    def _object2proto(self) -> GridHTTPConnection_PB:
        return GridHTTPConnection_PB(
            base_url=str(self.base_url),
            session_token=self.session_token,
            token_type=self.token_type,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GridHTTPConnection_PB
