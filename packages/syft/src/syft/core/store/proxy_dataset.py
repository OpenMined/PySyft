# stdlib
from typing import Any
from typing import Tuple

# relative
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serializable import serializable
from ...core.common.uid import UID
from .util import get_s3_client


@serializable(recursive_serde=True)
class ProxyDataClass:
    __attr_allowlist__ = ["node_id", "asset_name", "dataset_name", "shape", "dtype"]

    def __init__(
        self,
        asset_name: str,
        dataset_name: str,
        shape: Tuple[int, ...],
        dtype: str,
        node_id: UID,
    ) -> None:
        self.asset_name = asset_name
        self.dataset_name = dataset_name
        self.shape = shape
        self.dtype = dtype
        self.node_id = node_id

    @property
    def name(self) -> str:
        return self.dataset_name + "/" + self.asset_name

    def get_s3_data(self) -> Any:
        # response = node.datasets.perform_request(
        #     syft_msg=DownloadDataMessage,
        #     context={"filename": self.name}
        # )
        # download_url = response.payload.url
        s3_client = get_s3_client(docker_host=True)
        if s3_client is None:
            return
        response = s3_client.get_object(Bucket=self.node_id.no_dash, Key=self.name)
        data = response.get("Body", b"").read()
        # msg = DownloadDataMessage(address=node.address, kwargs={"filename": self.name}, reply_to=node.address)
        # download_url = msg.run(node, verify_key).url
        return deserialize(data, from_bytes=True)
