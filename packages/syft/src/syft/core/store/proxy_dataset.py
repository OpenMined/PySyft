# stdlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

# third party
from pydantic import BaseSettings

# relative
from ...core.common.serde.deserialize import _deserialize as deserialize
from ...core.common.serde.serializable import serializable
from ...core.common.uid import UID
from ...grid import GridURL


@serializable(recursive_serde=True)
class ProxyDataset:
    __attr_allowlist__ = [
        "node_id",
        "asset_name",
        "dataset_name",
        "shape",
        "dtype",
        "fqn",
        "url",
        "obj_public_kwargs",
    ]

    def __init__(
        self,
        asset_name: str,
        dataset_name: str,
        shape: Tuple[int, ...],
        dtype: str,
        fqn: str,
        node_id: UID,
        obj_public_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.asset_name = asset_name
        self.dataset_name = dataset_name
        self.shape = shape
        self.dtype = dtype
        self.fqn = fqn
        self.node_id = node_id
        self.url = ""
        self.obj_public_kwargs = (
            obj_public_kwargs if obj_public_kwargs is not None else {}
        )

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__}: {self.dataset_name}[{self.asset_name}] -> "
            + f"shape: {self.shape} dtype: {self.dtype} url: {self.url}>"
        )

    @property
    def name(self) -> str:
        return self.dataset_name + "/" + self.asset_name

    @property
    def data_fully_qualified_name(self) -> str:
        return self.fqn

    def get_s3_data(self, settings: BaseSettings) -> Any:
        try:
            # relative
            from ..node.common.util import get_s3_client

            s3_client = get_s3_client(settings=settings)
            if s3_client is None:
                raise Exception("get_s3_client returned None")
            response = s3_client.get_object(Bucket=self.node_id.no_dash, Key=self.name)
            data = response.get("Body", b"").read()
            return deserialize(data, from_bytes=True)
        except Exception as e:
            print(f"Failed to get data from proxy object {e}.")
            raise e

    def generate_presigned_url(
        self, settings: BaseSettings, public_url: bool = False
    ) -> None:
        # relative
        from ..node.common.util import get_s3_client

        s3_client = get_s3_client(settings=settings)

        download_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.node_id.no_dash, "Key": self.name},
            ExpiresIn=settings.S3_PRESIGNED_TIMEOUT_SECS,
            HttpMethod="GET",
        )

        if public_url:
            grid_url = GridURL.from_url(url=download_url)
            # add /blob to path
            grid_url.path = f"/blob{grid_url.path}"
            download_url = grid_url.url_path

        self.url = download_url
