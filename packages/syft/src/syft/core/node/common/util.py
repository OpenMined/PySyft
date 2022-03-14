# stdlib
from io import BytesIO
from typing import Any
from typing import Generator
from typing import List

# relative
from ...common.serde.serialize import _serialize as serialize
from ...common.uid import UID
from ...store.proxy_dataset import ProxyDataClass
from ...store.util import get_s3_client


def read_chunks(
    fp: BytesIO, chunk_size: int = 1024**3
) -> Generator[bytes, None, None]:
    """Read data in chunks from the file."""
    while True:
        data = fp.read(chunk_size)
        if not data:
            break
        yield data


def listify(x: Any) -> List[Any]:
    """turns x into a list.
    If x is a list or tuple, return as list.
    if x is not a list: return [x]
    if x is None: return []

    Args:
        x (Any): some object

    Returns:
        List[Any]: x, as a list
    """
    return list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [x])


def upload_result_to_s3(
    asset_name: str, dataset_name: str, domain_id: UID, data: Any
) -> ProxyDataClass:

    s3_client = get_s3_client(docker_host=True)

    binary_dataset: bytes = serialize(data, to_bytes=True)  # type: ignore

    # 1 - Starts to upload binary data into Seaweed.
    # TODO: Make this a resumable upload and ADD progress bar.
    binary_buffer = BytesIO(binary_dataset)

    filename = f"{dataset_name}/{asset_name}"

    # 3 - Send a message to PyGrid warning about dataset upload complete!
    upload_response = s3_client.put_object(
        Bucket=domain_id.no_dash,
        Body=binary_buffer,
        Key=filename,
        ContentType="application/octet-stream",
    )

    # TODO: Throw an exception if the response is not valid
    print("Upload Result")
    print(upload_response)

    data_dtype = str(type(data))
    proxy_obj = ProxyDataClass(
        asset_name=asset_name,
        dataset_name=dataset_name,
        node_id=domain_id,
        dtype=data_dtype,
        shape=data.shape,
    )
    return proxy_obj
