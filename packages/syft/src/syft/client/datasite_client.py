# future
from __future__ import annotations

# stdlib
import json
import logging
from pathlib import Path
import re
from string import Template
import traceback
from typing import TYPE_CHECKING
from typing import cast

# third party
import markdown
from tqdm import tqdm

# relative
from ..abstract_server import ServerSideType
from ..serde.serializable import serializable
from ..service.action.action_object import ActionObject
from ..service.code_history.code_history import CodeHistoriesDict
from ..service.code_history.code_history import UsersCodeHistoriesDict
from ..service.dataset.dataset import Contributor
from ..service.dataset.dataset import CreateAsset
from ..service.dataset.dataset import CreateDataset
from ..service.dataset.dataset import _check_asset_must_contain_mock
from ..service.migration.object_migration_state import MigrationData
from ..service.request.request import Request
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..service.sync.diff_state import ResolvedSyncState
from ..service.sync.sync_state import SyncState
from ..service.user.roles import Roles
from ..service.user.user import ServiceRole
from ..service.user.user import UserView
from ..types.blob_storage import BlobFile
from ..types.errors import SyftException
from ..types.uid import UID
from ..util.misc_objs import HTMLObject
from ..util.util import get_mb_size
from ..util.util import prompt_warning_message
from .api import APIModule
from .client import SyftClient
from .client import login
from .client import login_as_guest
from .connection import ServerConnection
from .protocol import SyftProtocol

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # relative
    from ..orchestra import ServerHandle
    from ..service.project.project import Project


def _get_files_from_glob(glob_path: str) -> list[Path]:
    files = Path().glob(glob_path)
    return [f for f in files if f.is_file() and not f.name.startswith(".")]


def _get_files_from_dir(dir: Path, recursive: bool) -> list:
    files = dir.rglob("*") if recursive else dir.iterdir()
    return [f for f in files if not f.name.startswith(".") and f.is_file()]


def _contains_subdir(dir: Path) -> bool:
    for item in dir.iterdir():
        if item.is_dir():
            return True
    return False


def add_default_uploader(
    user: UserView, obj: CreateDataset | CreateAsset
) -> CreateDataset | CreateAsset:
    uploader = None
    for contributor in obj.contributors:
        if contributor.role == str(Roles.UPLOADER):
            uploader = contributor
            break

    if uploader is None:
        uploader = Contributor(
            role=str(Roles.UPLOADER),
            name=user.name,
            email=user.email,
        )
        obj.contributors.add(uploader)

    obj.uploader = uploader
    return obj


@serializable(canonical_name="DatasiteClient", version=1)
class DatasiteClient(SyftClient):
    def __repr__(self) -> str:
        return f"<DatasiteClient: {self.name}>"

    def upload_dataset(self, dataset: CreateDataset) -> SyftSuccess:
        # relative
        from ..types.twin_object import TwinObject

        if self.users is None:
            raise SyftException(public_message=f"can't get user service for {self}")

        user = self.users.get_current_user()
        if user.role not in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]:
            return SyftError(message="You don't have permission to upload datasets.")
        dataset = add_default_uploader(user, dataset)

        for i in range(len(dataset.asset_list)):
            asset = dataset.asset_list[i]
            dataset.asset_list[i] = add_default_uploader(user, asset)

        # dataset._check_asset_must_contain_mock()
        dataset_size: float = 0.0

        # TODO: Refactor so that object can also be passed to generate warnings

        self.api.connection = cast(ServerConnection, self.api.connection)

        metadata = self.api.connection.get_server_metadata(self.api.signing_key)

        if (
            metadata.show_warnings
            and metadata.server_side_type == ServerSideType.HIGH_SIDE.value
        ):
            message = (
                "You're approving a request on "
                f"{metadata.server_side_type} side {metadata.server_type} "
                "which may host datasets with private information."
            )
            prompt_warning_message(message=message, confirm=True)

        with tqdm(
            total=len(dataset.asset_list), colour="green", desc="Uploading"
        ) as pbar:
            for asset in dataset.asset_list:
                try:
                    contains_empty: bool = asset.contains_empty()
                    twin = TwinObject(
                        private_obj=ActionObject.from_obj(asset.data),
                        mock_obj=ActionObject.from_obj(asset.mock),
                        syft_server_location=self.id,
                        syft_client_verify_key=self.verify_key,
                    )
                    twin._save_to_blob_storage(allow_empty=contains_empty).unwrap()
                except Exception as e:
                    tqdm.write(f"Failed to create twin for {asset.name}. {e}")
                    raise SyftException(
                        public_message=f"Failed to create twin. {e}"
                    ) from e

                try:
                    self.api.services.action.set(
                        twin, ignore_detached_objs=contains_empty
                    )
                except SyftException:
                    tqdm.write(f"Failed to upload asset: {asset.name}")
                    raise

                asset.action_id = twin.id
                asset.server_uid = self.id
                dataset_size += get_mb_size(asset.data)

                # Update the progress bar and set the dynamic description
                pbar.set_description(f"Uploading: {asset.name}")
                pbar.update(1)

        dataset.mb_size = dataset_size
        _check_asset_must_contain_mock(dataset.asset_list)
        dataset.check()
        return self.api.services.dataset.add(dataset=dataset)

    def forgot_password(self, email: str) -> SyftSuccess | SyftError:
        return self.connection.forgot_password(email=email)

    def reset_password(self, token: str, new_password: str) -> SyftSuccess | SyftError:
        return self.connection.reset_password(token=token, new_password=new_password)

    def refresh(self) -> None:
        if self.credentials:
            self._fetch_server_metadata(self.credentials)

        if self._api and self._api.refresh_api_callback:
            self._api.refresh_api_callback()

    def get_sync_state(self) -> SyncState:
        state: SyncState = self.api.services.sync._get_state()
        for uid, obj in state.objects.items():
            if isinstance(obj, ActionObject):
                obj = obj.refresh_object(resolve_nested=False)
                state.objects[uid] = obj
        return state

    def apply_state(self, resolved_state: ResolvedSyncState) -> SyftSuccess:
        if len(resolved_state.delete_objs):
            prompt_warning_message(
                message=(
                    "Attempted to delete objects by syncing. "
                    "This is not currently supported, objects must be deleted manually."
                )
            )
        items = resolved_state.create_objs + resolved_state.update_objs

        action_objects = [x for x in items if isinstance(x, ActionObject)]
        for action_object in action_objects:
            action_object.reload_cache()
            # NOTE permissions are added separately server side
            action_object._send(self.id, self.verify_key, add_storage_permission=False)
            action_object._clear_cache()

        ignored_batches = resolved_state.ignored_batches

        res = self.api.services.sync.sync_items(
            items,
            resolved_state.new_permissions,
            resolved_state.new_storage_permissions,
            ignored_batches,
            unignored_batches=resolved_state.unignored_batches,
        )
        self._fetch_api(self.credentials)
        return res

    def upload_files(
        self,
        file_list: BlobFile | list[BlobFile] | str | list[str] | Path | list[Path],
        allow_recursive: bool = False,
        show_files: bool = False,
    ) -> SyftSuccess:
        if not file_list:
            raise SyftException(public_message="No files to upload")

        if not isinstance(file_list, list):
            file_list = [file_list]  # type: ignore[assignment]
        file_list = cast(list, file_list)

        expanded_file_list: list[BlobFile | Path] = []

        for file in file_list:
            if isinstance(file, BlobFile):
                expanded_file_list.append(file)
                continue

            path = Path(file)

            if re.search(r"[\*\?\[]", str(path)):
                expanded_file_list.extend(_get_files_from_glob(str(path)))
            elif path.is_dir():
                if not allow_recursive and _contains_subdir(path):
                    res = input(
                        f"Do you want to include all files recursively in {path.absolute()}? [y/n]: "
                    ).lower()
                    print(
                        f'{"Recursively uploading all files" if res == "y" else "Uploading files"} in {path.absolute()}'
                    )
                    allow_recursive = res == "y"
                expanded_file_list.extend(_get_files_from_dir(path, allow_recursive))
            elif path.exists():
                expanded_file_list.append(path)

        if not expanded_file_list:
            raise SyftException(public_message="No files to upload were found")

        print(
            f"Uploading {len(expanded_file_list)} {'file' if len(expanded_file_list) == 1 else 'files'}:"
        )

        if show_files:
            for file in expanded_file_list:
                if isinstance(file, BlobFile):
                    print(file.path or file.file_name)
                else:
                    print(file.absolute())

        try:
            result = []
            for file in expanded_file_list:
                if not isinstance(file, BlobFile):
                    file = BlobFile(path=file, file_name=file.name)
                print("Uploading", file.file_name)
                if not file.uploaded:
                    file.upload_to_blobstorage(self)
                result.append(file)

            return ActionObject.from_obj(result).send(self)
        except Exception as err:
            raise SyftException(
                public_message=f"Failed to upload files: {err}.\n{traceback.format_exc()}"
            )

    def connect_to_gateway(
        self,
        via_client: SyftClient | None = None,
        url: str | None = None,
        port: int | None = None,
        handle: ServerHandle | None = None,  # noqa: F821
        email: str | None = None,
        password: str | None = None,
        protocol: str | SyftProtocol = SyftProtocol.HTTP,
        reverse_tunnel: bool = False,
    ) -> SyftSuccess | None:
        if isinstance(protocol, str):
            protocol = SyftProtocol(protocol)

        if via_client is not None:
            client = via_client
        elif handle is not None:
            client = handle.client
        else:
            client = (
                login_as_guest(url=url, port=port)
                if email is None
                else login(url=url, port=port, email=email, password=password)
            )

        res = self.exchange_route(
            client,
            protocol=protocol,
            reverse_tunnel=reverse_tunnel,
        )
        if isinstance(res.value, Request):
            return res.value
        else:
            if self.metadata:
                return SyftSuccess(
                    message=(
                        f"Connected {self.metadata.server_type} "
                        f"'{self.metadata.name}' to gateway '{client.name}'. "
                        f"{res.message}"
                    )
                )
            else:
                return SyftSuccess(message=f"Connected to '{client.name}' gateway")

    def _get_service_by_name_if_exists(self, name: str) -> APIModule | None:
        if self.api.has_service(name):
            return getattr(self.api.services, name)
        return None

    def set_server_side_type_dangerous(self, server_side_type: str) -> SyftSuccess:
        return self.api.services.settings.set_server_side_type_dangerous(
            server_side_type
        )

    @property
    def data_subject_registry(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("data_subject")

    @property
    def code(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("code")

    @property
    def worker(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("worker")

    @property
    def requests(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("request")

    @property
    def datasets(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("dataset")

    @property
    def projects(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("project")

    @property
    def code_history_service(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("code_history")

    @property
    def code_history(self) -> CodeHistoriesDict:
        return self.api.services.code_history.get_history()

    @property
    def code_histories(self) -> UsersCodeHistoriesDict:
        return self.api.services.code_history.get_histories()

    @property
    def images(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("worker_image")

    @property
    def worker_pools(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("worker_pool")

    @property
    def worker_images(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("worker_images")

    @property
    def sync(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("sync")

    @property
    def code_status(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("code_status")

    @property
    def output(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("output")

    @property
    def migration(self) -> APIModule | None:
        return self._get_service_by_name_if_exists("migration")

    def get_migration_data(self, include_blobs: bool = True) -> MigrationData:
        res = self.api.services.migration.get_migration_data()
        if include_blobs:
            res.download_blobs()

        return res

    def load_migration_data(
        self,
        path_or_data: str | Path | MigrationData,
        include_worker_pools: bool = False,
        with_reset_db: bool = False,
    ) -> SyftSuccess:
        if isinstance(path_or_data, MigrationData):
            migration_data = path_or_data
        else:
            migration_data = MigrationData.from_file(path_or_data)

        migration_data._set_obj_location_(self.id, self.verify_key)

        if self.id != migration_data.server_uid:
            raise SyftException(
                public_message=f"This Migration data is not for this server. Expected server id {self.id}, "
                f"got {migration_data.server_uid}"
            )

        if migration_data.signing_key.verify_key != self.verify_key:
            raise SyftException(
                public_message="Root verify key in migration data does not match this client's verify key"
            )

        if migration_data.includes_custom_workerpools and not include_worker_pools:
            prompt_warning_message(
                "This migration data includes custom workers, "
                "which need to be migrated separately with `sy.upgrade_custom_workerpools` "
                "after finishing the migration."
            )

        migration_data.migrate_and_upload_blobs()
        migration_data = migration_data.copy_without_blobs()

        if not include_worker_pools:
            migration_data = migration_data.copy_without_workerpools()

        if with_reset_db:
            return self.api.services.migration.reset_and_restore(migration_data)
        else:
            return self.api.services.migration.apply_migration_data(migration_data)

    def dump_state(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        path.mkdir(exist_ok=True)
        blob_path = path / "migration.blob"
        yaml_path = path / "migration.yaml"
        config_path = path / "config.json"

        migration_data = self.get_migration_data(include_blobs=True)
        migration_data.save(blob_path, yaml_path=yaml_path)
        server_config = self.api.services.settings.get_server_config()
        with open(config_path, "w") as fp:
            json.dump(server_config, fp)

    def get_project(
        self,
        name: str | None = None,
        uid: UID | None = None,
    ) -> Project | None:
        """Get project by name or UID"""

        if not self.api.has_service("project"):
            return None

        if name:
            return self.api.services.project.get_by_name(name)

        elif uid:
            return self.api.services.project.get_by_uid(uid)

        return self.api.services.project.get_all()

    def _repr_html_(self) -> str:
        obj = self.api.services.settings.welcome_show()
        updated_template_str = Template(obj.text).safe_substitute(
            server_url=getattr(self.connection, "url", None)
        )
        # If it's a markdown structured file
        if not isinstance(obj, HTMLObject):
            return markdown.markdown(updated_template_str)

        # if it's a html string
        return updated_template_str
