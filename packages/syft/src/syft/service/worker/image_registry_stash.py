# stdlib
from typing import Literal

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.errors import SyftException
from ...types.result import as_result
from .image_registry import SyftImageRegistry


@serializable(canonical_name="SyftImageRegistrySQLStash", version=1)
class SyftImageRegistryStash(ObjectStash[SyftImageRegistry]):
    @as_result(SyftException, StashException, NotFoundException)
    def get_by_url(
        self,
        credentials: SyftVerifyKey,
        url: str,
    ) -> SyftImageRegistry:
        return self.get_one(
            credentials=credentials,
            filters={"url": url},
        ).unwrap()

    @as_result(SyftException, StashException)
    def delete_by_url(self, credentials: SyftVerifyKey, url: str) -> Literal[True]:
        item = self.get_by_url(credentials=credentials, url=url).unwrap()
        self.delete_by_uid(credentials=credentials, uid=item.id).unwrap()

        # TODO standardize delete return type
        return True
