# stdlib
import io
import json
import sys
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

# third party
import requests
from requests_toolbelt.multipart import encoder

# syft relative
from ...core.common.message import SyftMessage
from ...core.common.serde.serialize import _serialize
from ...proto.core.node.common.metadata_pb2 import Metadata as Metadata_PB
from ..client.enums import RequestAPIFields
from ..client.exceptions import RequestAPIException
from ..connections.http_connection import HTTPConnection


class GridHTTPConnection(HTTPConnection):
    LOGIN_ROUTE = "/users/login"
    SYFT_ROUTE = "/pysyft"
    SYFT_MULTIPART_ROUTE = "/pysyft_multipart"
    SIZE_THRESHOLD = 20971520  # 20 MB

    def __init__(self, url: str) -> None:
        self.base_url = url
        self.session_token: Optional[Dict[str, str]] = None

    def _send_msg(self, msg: SyftMessage) -> requests.Response:
        """
        Serializes Syft messages in json format and send it using HTTP protocol.
        NOTE: Auxiliary method to avoid code duplication and modularity.
        :return: returns requests.Response object containing a JSON serialized
        SyftMessage
        :rtype: requests.Response
        """

        header = {}

        if self.session_token:
            header["token"] = self.session_token

        header["Content-Type"] = "application/octet-stream"  # type: ignore

        # Perform HTTP request using base_url as a root address
        msg_bytes: bytes = _serialize(obj=msg, to_bytes=True)  # type: ignore

        if sys.getsizeof(msg_bytes) < GridHTTPConnection.SIZE_THRESHOLD:
            r = requests.post(
                url=self.base_url + GridHTTPConnection.SYFT_ROUTE,
                data=msg_bytes,
                headers=header,
            )
        else:
            r = self.send_streamed_messages(blob_message=msg_bytes)

        # Return request's response object
        # r.text provides the response body as a str
        return r

    def login(self, credentials: Dict) -> Tuple:
        # Login request
        response = requests.post(
            url=self.base_url + GridHTTPConnection.LOGIN_ROUTE, json=credentials
        )

        # Response
        content = json.loads(response.text)

        # If fail
        if response.status_code != requests.codes.ok:
            raise Exception(content["error"])

        metadata = content["metadata"].encode("ISO-8859-1")
        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(metadata)

        # If success
        # Save session token
        self.session_token = content["token"]

        # Return node metadata / user private key
        return (metadata_pb, content["key"])

    def _get_metadata(self) -> Tuple:
        """Request Node's metadata
        :return: returns node metadata
        :rtype: str of bytes
        """
        response = requests.get(self.base_url + "/metadata")
        content = json.loads(response.text)

        metadata = content["metadata"].encode("ISO-8859-1")
        metadata_pb = Metadata_PB()
        metadata_pb.ParseFromString(metadata)

        return metadata_pb

    def setup(self, **content: Dict[str, Any]) -> Any:
        response = json.loads(
            requests.post(self.base_url + "/setup", json=content).text
        )
        if response.get(RequestAPIFields.MESSAGE, None):
            return response
        else:
            raise RequestAPIException(response.get(RequestAPIFields.ERROR))

    def send_files(self, file_path: str) -> Dict[str, Any]:
        session = requests.Session()

        with open(file_path, "rb") as f:

            form = encoder.MultipartEncoder(
                {
                    "file": (file_path, f, "application/octet-stream"),
                }
            )

            headers = {
                "Prefer": "respond-async",
                "Content-Type": form.content_type,
                "token": self.session_token,
            }

            resp = session.post(
                self.base_url + "/data-centric/datasets", headers=headers, data=form
            )

        session.close()

        return json.loads(resp.content)

    def send_streamed_messages(self, blob_message: bytes) -> requests.Response:
        session = requests.Session()
        with io.BytesIO(blob_message) as msg:
            form = encoder.MultipartEncoder(
                {
                    "file": ("message", msg.read(), "application/octet-stream"),
                }
            )

            headers = {
                "Prefer": "respond-async",
                "Content-Type": form.content_type,
            }

            resp = session.post(
                self.base_url + GridHTTPConnection.SYFT_MULTIPART_ROUTE,
                headers=headers,
                data=form,
            )

        session.close()
        return resp
