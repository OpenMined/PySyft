# stdlib
from io import BytesIO
import math
from pathlib import Path
from typing import Generator
from typing import List
from typing import Optional
from typing import Type
from typing import Union

# third party
import boto3
from botocore.client import BaseClient as S3BaseClient
from botocore.client import ClientError as BotoClientError
from botocore.client import Config
import requests
from typing_extensions import Self

# relative
from . import BlobDeposit
from . import BlobRetrieval
from . import BlobRetrievalByURL
from . import BlobStorageClient
from . import BlobStorageClientConfig
from . import BlobStorageConfig
from . import BlobStorageConnection
from ...serde.serializable import serializable
from ...service.response import SyftError
from ...service.response import SyftException
from ...service.response import SyftSuccess
from ...service.service import from_api_or_context
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SeaweedSecureFilePathLocation
from ...types.blob_storage import SecureFilePathLocation
from ...types.grid_url import GridURL
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...util.constants import DEFAULT_TIMEOUT

READ_EXPIRATION_TIME = 1800  # seconds
WRITE_EXPIRATION_TIME = 900  # seconds
DEFAULT_CHUNK_SIZE = 1024**3  # 1 GB


def _byte_chunks(bytes: BytesIO, size: int) -> Generator[bytes, None, None]:
    while True:
        try:
            yield bytes.read(size)
        except BlockingIOError:
            return


@serializable()
class SeaweedFSBlobDeposit(BlobDeposit):
    __canonical_name__ = "SeaweedFSBlobDeposit"
    __version__ = SYFT_OBJECT_VERSION_1

    urls: List[GridURL]

    def write(self, data: BytesIO) -> Union[SyftSuccess, SyftError]:
        # relative
        from ...client.api import APIRegistry

        api = APIRegistry.api_for(
            node_uid=self.syft_node_location,
            user_verify_key=self.syft_client_verify_key,
        )

        etags = []

        try:
            for part_no, (byte_chunk, url) in enumerate(
                zip(_byte_chunks(data, DEFAULT_CHUNK_SIZE), self.urls),
                start=1,
            ):
                if api is not None:
                    blob_url = api.connection.to_blob_route(url.url_path)
                else:
                    blob_url = url
                response = requests.put(
                    url=str(blob_url), data=byte_chunk, timeout=DEFAULT_TIMEOUT
                )
                response.raise_for_status()
                etag = response.headers["ETag"]
                etags.append({"ETag": etag, "PartNumber": part_no})
        except requests.RequestException as e:
            return SyftError(message=str(e))

        mark_write_complete_method = from_api_or_context(
            func_or_path="blob_storage.mark_write_complete",
            syft_node_location=self.syft_node_location,
            syft_client_verify_key=self.syft_client_verify_key,
        )
        return mark_write_complete_method(
            etags=etags,
            uid=self.blob_storage_entry_id,
        )


@serializable()
class SeaweedFSClientConfig(BlobStorageClientConfig):
    host: str
    port: int
    access_key: str
    secret_key: str
    region: str
    bucket_name: str

    @property
    def endpoint_url(self) -> str:
        grid_url = GridURL(host_or_ip=self.host, port=self.port)
        return grid_url.url


@serializable()
class SeaweedFSClient(BlobStorageClient):
    config: SeaweedFSClientConfig

    def connect(self) -> BlobStorageConnection:
        return SeaweedFSConnection(
            client=boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                config=Config(signature_version="s3v4"),
                region_name=self.config.region,
            ),
            bucket_name=self.config.bucket_name,
        )


@serializable()
class SeaweedFSConnection(BlobStorageConnection):
    client: S3BaseClient
    bucket_name: str

    def __init__(self, client: S3BaseClient, bucket_name: str):
        self.client = client
        self.bucket_name = bucket_name

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.client.close()

    def read(self, fp: SecureFilePathLocation, type_: Optional[Type]) -> BlobRetrieval:
        try:
            url = self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket_name, "Key": fp.path},
                ExpiresIn=READ_EXPIRATION_TIME,
            )

            return BlobRetrievalByURL(
                url=GridURL.from_url(url), file_name=Path(fp.path).name, type_=type_
            )
        except BotoClientError as e:
            raise SyftException(e)

    def allocate(
        self, obj: CreateBlobStorageEntry
    ) -> Union[SecureFilePathLocation, SyftError]:
        try:
            file_name = obj.file_name
            result = self.client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=file_name,
            )
            upload_id = result["UploadId"]
            return SeaweedSecureFilePathLocation(upload_id=upload_id, path=file_name)
        except BotoClientError as e:
            return SyftError(
                message=f"Failed to allocate space for {obj} with error: {e}"
            )

    def write(self, obj: BlobStorageEntry) -> BlobDeposit:
        total_parts = math.ceil(obj.file_size / DEFAULT_CHUNK_SIZE)

        urls = [
            GridURL.from_url(
                self.client.generate_presigned_url(
                    ClientMethod="upload_part",
                    Params={
                        "Bucket": self.bucket_name,
                        "Key": obj.location.path,
                        "UploadId": obj.location.upload_id,
                        "PartNumber": i + 1,
                    },
                    ExpiresIn=WRITE_EXPIRATION_TIME,
                )
            )
            for i in range(total_parts)
        ]

        return SeaweedFSBlobDeposit(blob_storage_entry_id=obj.id, urls=urls)

    def complete_multipart_upload(
        self,
        blob_entry: BlobStorageEntry,
        etags: List,
    ) -> Union[SyftError, SyftSuccess]:
        try:
            self.client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=blob_entry.location.path,
                MultipartUpload={"Parts": etags},
                UploadId=blob_entry.location.upload_id,
            )
            return SyftSuccess(message="Successfully saved file.")
        except BotoClientError as e:
            return SyftError(message=str(e))

    def delete(
        self,
        fp: SecureFilePathLocation,
    ) -> Union[SyftSuccess, SyftError]:
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=fp.path)
            return SyftSuccess(message="Successfully deleted file.")
        except BotoClientError as e:
            return SyftError(message=str(e))


@serializable()
class SeaweedFSConfig(BlobStorageConfig):
    client_type: Type[BlobStorageClient] = SeaweedFSClient
    client_config: SeaweedFSClientConfig
