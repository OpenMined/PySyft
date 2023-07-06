# future
from __future__ import annotations

# stdlib
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

# third party
from tqdm import tqdm
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..service.dataset.dataset import CreateDataset
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.uid import UID
from ..util.util import get_mb_size
from .api import APIModule
from .client import SyftClient

if TYPE_CHECKING:
    # relative
    from ..service.project.project import Project


@serializable()
class DomainClient(SyftClient):
    def __repr__(self) -> str:
        return f"<DomainClient: {self.name}>"

    def upload_dataset(self, dataset: CreateDataset) -> Union[SyftSuccess, SyftError]:
        # relative
        from ..types.twin_object import TwinObject

        dataset._check_asset_must_contain_mock()
        dataset_size = 0

        for asset in tqdm(dataset.asset_list):
            print(f"Uploading: {asset.name}")
            try:
                twin = TwinObject(private_obj=asset.data, mock_obj=asset.mock)
            except Exception as e:
                return SyftError(message=f"Failed to create twin. {e}")
            response = self.api.services.action.set(twin)
            if isinstance(response, SyftError):
                print(f"Failed to upload asset\n: {asset}")
                return response
            asset.action_id = twin.id
            asset.node_uid = self.id
            dataset_size += get_mb_size(asset.data)
        dataset.mb_size = dataset_size
        valid = dataset.check()
        if valid.ok():
            return self.api.services.dataset.add(dataset=dataset)
        else:
            if len(valid.err()) > 0:
                return tuple(valid.err())
            return valid.err()

    def apply_to_gateway(self, client: Self) -> None:
        return self.exchange_route(client)

    @property
    def data_subject_registry(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("data_subject"):
            return self.api.services.data_subject
        return None

    @property
    def code(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("code"):
            return self.api.services.code

    @property
    def requests(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("request"):
            return self.api.services.request
        return None

    @property
    def datasets(self) -> Optional[APIModule]:
        if self.api is not None and self.api.has_service("dataset"):
            return self.api.services.dataset
        return None

    @property
    def projects(self) -> Optional[APIModule]:
        if self.api.has_service("project"):
            return self.api.services.project
        return None

    def get_project(
        self,
        name: str = None,
        uid: UID = None,
    ) -> Optional[Project]:
        """Get project by name or UID"""

        if not self.api.has_service("project"):
            return None

        if name:
            return self.api.services.project.get_by_name(name)

        elif uid:
            return self.api.services.project.get_by_uid(uid)

        return self.api.services.project.get_all()
