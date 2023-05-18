# stdlib
from typing import List
from typing import Union

# relative
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..action.action_permissions import ActionObjectPermission
from ..action.action_permissions import ActionPermission
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import SERVICE_TO_TYPES
from ..service import TYPE_TO_SERVICE
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import GUEST_ROLE_LEVEL
from ...types.syft_file import SyftFile, CreateSyftFile
from .syft_file_stash import SyftFileStash


@instrument
@serializable()
class SyftFileService(AbstractService):
    store: DocumentStore
    stash: SyftFileStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = SyftFileStash(store=store)

    @service_method(path="syft_file.add", name="add", roles=DATA_OWNER_ROLE_LEVEL)
    def add(
        self, context: AuthedServiceContext, syft_file: CreateSyftFile
    ) -> Union[SyftSuccess, SyftError]:
        """Add a SyftFile"""
        # TODO: create a path in /tmp/syft
        # this would depend on the OS, probably this would need to be decided in another file
        # maybe more general so other parts of the codebase can access the "api"
        # TODO: auto create folder
        # context.output['path'] = '/tmp/syft'
        # syft_file = syft_file.to(SyftFile, context=context)
        CreateSyftFile.write_file(
            filename=syft_file.filename,
            data=syft_file.data,
            path='/tmp/syft'    
        )
        syft_file = SyftFile(
            id=UID(),
            filename=syft_file.filename,
            size_bytes=syft_file.size_bytes,
            mimetype=syft_file.mimetype
        )
        result = self.stash.set(
            context.credentials,
            syft_file,
            add_permissions=[
                ActionObjectPermission(
                    uid=syft_file.id, permission=ActionPermission.ALL_READ
                ),
            ],
        )
        if result.is_err():
            return SyftError(message=str(result.err()))
        return SyftSuccess(message="SyftFile Added")

    @service_method(path="syft_file.get_all", name="get_all", roles=GUEST_ROLE_LEVEL)
    def get_all(self, context: AuthedServiceContext) -> Union[List[SyftFile], SyftError]:
        """Get a SyftFile"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            syft_files = result.ok()
            results = []
            for syft_file in syft_files:
                results.append(syft_file)
            return results
        return SyftError(message=result.err())

    @service_method(path="syft_file.get_by_id", name="get_by_id")
    def get_by_id(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get a SyftFile"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            syft_file = result.ok()
            syft_file.node_uid = context.node.id
            return syft_file
        return SyftError(message=result.err())



TYPE_TO_SERVICE[SyftFile] = SyftFileService
SERVICE_TO_TYPES[SyftFileService].update({SyftFile})
