# stdlib

# third party

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.stash import ObjectStash
from ...store.document_store_errors import NotFoundException
from ...store.document_store_errors import StashException
from ...types.result import as_result
from ...util.telemetry import instrument
from .project import Project


@instrument
@serializable(canonical_name="ProjectSQLStash", version=1)
class ProjectStash(ObjectStash[Project]):
    @as_result(StashException)
    def get_all_for_verify_key(
        self, credentials: SyftVerifyKey, verify_key: SyftVerifyKey
    ) -> list[Project]:
        return self.get_all_by_field(
            credentials=credentials,
            field_name="user_verify_key",
            field_value=str(verify_key),
        ).unwrap()

    @as_result(StashException, NotFoundException)
    def get_by_name(self, credentials: SyftVerifyKey, project_name: str) -> Project:
        return self.get_one_by_field(
            credentials=credentials,
            field_name="name",
            field_value=project_name,
        ).unwrap()
