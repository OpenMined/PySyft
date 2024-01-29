# stdlib
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

# third party
import requests

# relative
from ...serde.serializable import serializable
from ...service.action.action_object import ActionObject
from ...store.blob_storage import BlobRetrieval
from ...store.blob_storage.on_disk import OnDiskBlobDeposit
from ...store.blob_storage.seaweedfs import SeaweedFSBlobDeposit
from ...store.document_store import DocumentStore
from ...store.document_store import UIDPartitionKey
from ...types.blob_storage import AzureSecureFilePathLocation
from ...types.blob_storage import BlobFileType
from ...types.blob_storage import BlobStorageEntry
from ...types.blob_storage import BlobStorageMetadata
from ...types.blob_storage import CreateBlobStorageEntry
from ...types.blob_storage import SeaweedSecureFilePathLocation
from ...types.uid import UID
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import GUEST_ROLE_LEVEL
from .remote_profile import AzureRemoteProfile
from .remote_profile import RemoteProfileStash
from .stash import BlobStorageStash

BlobDepositType = Union[OnDiskBlobDeposit, SeaweedFSBlobDeposit]


@serializable()
class BlobStorageService(AbstractService):
    store: DocumentStore
    stash: BlobStorageStash
    remote_profile_stash: RemoteProfileStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = BlobStorageStash(store=store)
        self.remote_profile_stash = RemoteProfileStash(store=store)

    @service_method(path="blob_storage.get_all", name="get_all")
    def get_all_blob_storage_entries(
        self, context: AuthedServiceContext
    ) -> Union[List[BlobStorageEntry], SyftError]:
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="blob_storage.mount_azure", name="mount_azure")
    def mount_azure(
        self,
        context: AuthedServiceContext,
        account_name: str,
        account_key: str,
        container_name: str,
        bucket_name: str,
        use_direct_connections=True,
    ):
        # stdlib

        # TODO: fix arguments

        remote_name = f"{account_name}{container_name}"
        remote_name = "".join(ch for ch in remote_name if ch.isalnum())
        args_dict = {
            "account_name": account_name,
            "account_key": account_key,
            "container_name": container_name,
            "remote_name": remote_name,
            "bucket_name": bucket_name,
        }

        new_profile = AzureRemoteProfile(
            profile_name=remote_name,
            account_name=account_name,
            account_key=account_key,
            container_name=container_name,
        )
        res = self.remote_profile_stash.set(context.credentials, new_profile)
        if res.is_err():
            return SyftError(message=res.value)
        remote_profile = res.ok()
        seaweed_config = context.node.blob_storage_client.config
        # we cache this here such that we can use it when reading a file from azure
        # from the remote_name
        seaweed_config.remote_profiles[remote_name] = remote_profile

        # TODO: possible wrap this in try catch
        cfg = context.node.blob_store_config.client_config
        init_request = requests.post(url=cfg.mount_url, json=args_dict)  # nosec
        print(init_request.content)
        # TODO check return code
        res = context.node.blob_storage_client.connect().client.list_objects(
            Bucket=bucket_name
        )
        # stdlib
        objects = res["Contents"]
        file_sizes = [object["Size"] for object in objects]
        file_paths = [object["Key"] for object in objects]
        if use_direct_connections:
            secure_file_paths = [
                AzureSecureFilePathLocation(
                    path=file_path,
                    azure_profile_name=remote_name,
                    bucket_name=bucket_name,
                )
                for file_path in file_paths
            ]
        else:
            secure_file_paths = [
                SeaweedSecureFilePathLocation(
                    path=file_path,
                )
                for file_path in file_paths
            ]

        for sfp, file_size in zip(secure_file_paths, file_sizes):
            blob_storage_entry = BlobStorageEntry(
                location=sfp,
                uploaded_by=context.credentials,
                file_size=file_size,
                type_=BlobFileType,
                bucket_name=bucket_name,
            )
            self.stash.set(context.credentials, blob_storage_entry)

        return SyftSuccess(message="Mounting Azure Successful!")

    @service_method(
        path="blob_storage.get_files_from_bucket", name="get_files_from_bucket"
    )
    def get_files_from_bucket(self, context: AuthedServiceContext, bucket_name: str):
        result = self.stash.find_all(context.credentials, bucket_name=bucket_name)
        if result.is_err():
            return result
        bse_list = result.ok()
        # stdlib

        blob_files = []
        for bse in bse_list:
            self.stash.set(obj=bse, credentials=context.credentials)
            # We create an empty ActionObject and set its blob_storage_entry_id to bse.id
            # such that we can call reload_cache which creates
            # the BlobRetrieval (user needs permission to do this)
            # This could be a BlobRetrievalByURL that creates a BlobFile
            # and then sets it in the cache (it does not contain the data, only the BlobFile).
            # In the client, when reading the file, we will creates **another**, blobretrieval
            # object to read the actual data
            blob_file = ActionObject.empty()
            blob_file.syft_blob_storage_entry_id = bse.id
            blob_file.syft_client_verify_key = context.credentials
            blob_file.syft_node_location = context.node.id
            blob_file.reload_cache()
            blob_files.append(blob_file.syft_action_data)

        return blob_files

    @service_method(path="blob_storage.get_by_uid", name="get_by_uid")
    def get_blob_storage_entry_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[BlobStorageEntry, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(path="blob_storage.get_metadata", name="get_metadata")
    def get_blob_storage_metadata_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[BlobStorageEntry, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            blob_storage_entry = result.ok()
            return blob_storage_entry.to(BlobStorageMetadata)
        return SyftError(message=result.err())

    # TODO: replace name with `create_blob_retrieval`
    @service_method(
        path="blob_storage.read",
        name="read",
        roles=GUEST_ROLE_LEVEL,
    )
    def read(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[BlobRetrieval, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            obj: BlobStorageEntry = result.ok()
            if obj is None:
                return SyftError(message=f"No blob storage entry exists for uid: {uid}")

            with context.node.blob_storage_client.connect() as conn:
                res: BlobRetrieval = conn.read(
                    obj.location, obj.type_, bucket_name=obj.bucket_name
                )
                res.syft_blob_storage_entry_id = uid
                res.file_size = obj.file_size
                return res
        return SyftError(message=result.err())

    @service_method(
        path="blob_storage.allocate",
        name="allocate",
        roles=GUEST_ROLE_LEVEL,
    )
    def allocate(
        self, context: AuthedServiceContext, obj: CreateBlobStorageEntry
    ) -> Union[BlobDepositType, SyftError]:
        with context.node.blob_storage_client.connect() as conn:
            secure_location = conn.allocate(obj)

            if isinstance(secure_location, SyftError):
                return secure_location

            blob_storage_entry = BlobStorageEntry(
                id=obj.id,
                location=secure_location,
                type_=obj.type_,
                mimetype=obj.mimetype,
                file_size=obj.file_size,
                uploaded_by=context.credentials,
            )
            blob_deposit = conn.write(blob_storage_entry)

        result = self.stash.set(context.credentials, blob_storage_entry)
        if result.is_err():
            return SyftError(message=f"{result.err()}")
        return blob_deposit

    @service_method(
        path="blob_storage.write_to_disk",
        name="write_to_disk",
        roles=GUEST_ROLE_LEVEL,
    )
    def write_to_disk(
        self, context: AuthedServiceContext, uid: UID, data: bytes
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(
            credentials=context.credentials,
            uid=uid,
        )
        if result.is_err():
            return SyftError(message=f"{result.err()}")

        obj: Optional[BlobStorageEntry] = result.ok()

        if obj is None:
            return SyftError(message=f"No blob storage entry exists for uid: {uid}")

        try:
            Path(obj.location.path).write_bytes(data)
            return SyftSuccess(message="File successfully saved.")
        except Exception as e:
            return SyftError(message=f"Failed to write object to disk: {e}")

    @service_method(
        path="blob_storage.mark_write_complete",
        name="mark_write_complete",
        roles=GUEST_ROLE_LEVEL,
    )
    def mark_write_complete(
        self,
        context: AuthedServiceContext,
        uid: UID,
        etags: List,
        no_lines: Optional[int] = 0,
    ) -> Union[SyftError, SyftSuccess]:
        result = self.stash.get_by_uid(
            credentials=context.credentials,
            uid=uid,
        )
        if result.is_err():
            return SyftError(message=f"{result.err()}")

        obj: Optional[BlobStorageEntry] = result.ok()

        if obj is None:
            return SyftError(message=f"No blob storage entry exists for uid: {uid}")

        obj.no_lines = no_lines
        result = self.stash.update(
            credentials=context.credentials,
            obj=obj,
        )
        if result.is_err():
            return SyftError(message=f"{result.err()}")

        with context.node.blob_storage_client.connect() as conn:
            result = conn.complete_multipart_upload(obj, etags)

        return result

    @service_method(path="blob_storage.delete", name="delete")
    def delete(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            obj = result.ok()

            if obj is None:
                return SyftError(message=f"No blob storage entry exists for uid: {uid}")
            try:
                with context.node.blob_storage_client.connect() as conn:
                    file_unlinked_result = conn.delete(obj.location)
            except Exception as e:
                return SyftError(message=f"Failed to delete file: {e}")

            if isinstance(file_unlinked_result, SyftError):
                return file_unlinked_result
            blob_storage_entry_deleted = self.stash.delete(
                context.credentials, UIDPartitionKey.with_obj(uid), has_permission=True
            )
            if blob_storage_entry_deleted.is_ok():
                return file_unlinked_result

        return SyftError(message=result.err())


TYPE_TO_SERVICE[BlobStorageEntry] = BlobStorageEntry
