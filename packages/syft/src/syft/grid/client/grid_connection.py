# stdlib
import io
import json
from typing import Any
from typing import Dict
from typing import Tuple

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests_toolbelt.multipart.encoder import MultipartEncoder

# relative
from ...core.common.message import ImmediateSyftMessageWithoutReply
from ...core.common.message import SignedImmediateSyftMessageWithoutReply
from ...core.common.message import SyftMessage
from ...core.common.serde.serializable import serializable
from ...core.common.serde.serialize import _serialize
from ...core.node.domain.enums import RequestAPIFields
from ...core.node.domain.exceptions import RequestAPIException
from ...proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB
from ...proto.grid.connections.http_connection_pb2 import (
    GridHTTPConnection as GridHTTPConnection_PB,
)
from ..connections.http_connection import HTTPConnection


@serializable()
class GridHTTPConnection(HTTPConnection):

    LOGIN_ROUTE = "/login"
    SYFT_ROUTE = "/syft"
    SYFT_ROUTE_STREAM = "/syft/stream"  # non blocking node
    # SYFT_MULTIPART_ROUTE = "/pysyft_multipart"
    SIZE_THRESHOLD = 20971520  # 20 MB

    def __init__(self, url: str) -> None:
        self.base_url = url
        self.session_token: str = ""
        self.token_type: str = "'"

    def _send_msg(self, msg: SyftMessage) -> requests.Response:
        """
        Serializes Syft messages in json format and send it using HTTP protocol.
        NOTE: Auxiliary method to avoid code duplication and modularity.
        :return: returns requests.Response object containing a JSON serialized
        SyftMessage
        :rtype: requests.Response
        """

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

        header["Content-Type"] = "application/octet-stream"

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

        # if sys.getsizeof(msg_bytes) < GridHTTPConnection.SIZE_THRESHOLD:
        # if True:
        r = requests.post(
            url=self.base_url + route,
            data=msg_bytes,
            headers=header,
        )
        # else:
        #     r = self.send_streamed_messages(blob_message=msg_bytes)

        # Return request's response object
        # r.text provides the response body as a str
        return r

    def login(self, credentials: Dict) -> Tuple:
        response = requests.post(
            url=self.base_url + GridHTTPConnection.LOGIN_ROUTE,
            json=credentials,
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

    def _get_metadata(self) -> Tuple:
        """Request Node's metadata
        :return: returns node metadata
        :rtype: str of bytes
        """
        # allow retry when connecting in CI
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.get(self.base_url + "/syft/metadata")
        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(response.content)

        return metadata_pb

    def setup(self, **content: Dict[str, Any]) -> Any:
        response = json.loads(
            requests.post(self.base_url + "/setup", json=content).text
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
                self.base_url + GridHTTPConnection.SYFT_ROUTE, headers=header
            ).text
        )
        if response.get(RequestAPIFields.MESSAGE, None):
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

        resp = requests.post(self.base_url + route, files=files, headers=header)

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
                self.base_url + GridHTTPConnection.SYFT_ROUTE_STREAM,
                headers=headers,
                data=form,
            )

        session.close()
        return resp

    @property
    def host(self) -> str:
        return self.base_url.replace("/api/v1", "")

    @staticmethod
    def _proto2object(proto: GridHTTPConnection_PB) -> "GridHTTPConnection":
        obj = GridHTTPConnection(url=proto.base_url)
        obj.session_token = proto.session_token
        obj.token_type = proto.token_type
        return obj

    def _object2proto(self) -> GridHTTPConnection_PB:
        return GridHTTPConnection_PB(
            base_url=self.base_url,
            session_token=self.session_token,
            token_type=self.token_type,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return GridHTTPConnection_PB
