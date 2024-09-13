# stdlib

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...store.db.db import DBManager
from ...store.document_store_errors import NotFoundException
from ...types.uid import UID
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..context import AuthedServiceContext
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from ..user.user_roles import ServiceRole
from .code_history import CodeHistoriesDict
from .code_history import CodeHistory
from .code_history import CodeHistoryView
from .code_history import UsersCodeHistoriesDict
from .code_history_stash import CodeHistoryStash


@serializable(canonical_name="CodeHistoryService", version=1)
class CodeHistoryService(AbstractService):
    stash: CodeHistoryStash

    def __init__(self, store: DBManager) -> None:
        self.stash = CodeHistoryStash(store=store)

    @service_method(
        path="code_history.submit_version",
        name="submit_version",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
        unwrap_on_success=False,
    )
    def submit_version(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode | UserCode,
        comment: str | None = None,
    ) -> SyftSuccess:
        if isinstance(code, SubmitUserCode):
            code = context.server.services.user_code._submit(
                context=context, submit_code=code
            ).unwrap()

        try:
            code_history = self.stash.get_by_service_func_name_and_verify_key(
                credentials=context.credentials,
                service_func_name=code.service_func_name,
                user_verify_key=context.credentials,
            ).unwrap()
        except NotFoundException:
            code_history = CodeHistory(
                id=UID(),
                server_uid=context.server.id,
                user_verify_key=context.credentials,
                service_func_name=code.service_func_name,
            )
            self.stash.set(credentials=context.credentials, obj=code_history).unwrap()

        code_history.add_code(code=code, comment=comment)
        res = self.stash.update(
            credentials=context.credentials, obj=code_history
        ).unwrap()
        return SyftSuccess(message="Code version submit success", value=res)

    @service_method(
        path="code_history.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_all(self, context: AuthedServiceContext) -> list[CodeHistory]:
        """Get a Dataset"""
        return self.stash.get_all(context.credentials).unwrap()

    @service_method(
        path="code_history.get", name="get", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_code_by_uid(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        """Get a User Code Item"""
        return self.stash.get_by_uid(context.credentials, uid=uid).unwrap()

    @service_method(path="code_history.delete", name="delete", unwrap_on_success=False)
    def delete(self, context: AuthedServiceContext, uid: UID) -> SyftSuccess:
        res = self.stash.delete_by_uid(context.credentials, uid).unwrap()
        return SyftSuccess(message="Succesfully deleted", value=res)

    def fetch_histories_for_user(
        self, context: AuthedServiceContext, user_verify_key: SyftVerifyKey
    ) -> CodeHistoriesDict:
        if context.role in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]:
            code_histories = self.stash.get_by_verify_key(
                credentials=context.server.verify_key, user_verify_key=user_verify_key
            ).unwrap()
        else:
            code_histories = self.stash.get_by_verify_key(
                credentials=context.credentials, user_verify_key=user_verify_key
            ).unwrap()

        def get_code(uid: UID) -> UserCode:
            return context.server.services.user_code.stash.get_by_uid(
                credentials=context.server.verify_key,
                uid=uid,
            ).unwrap()

        code_versions_dict = {}

        for code_history in code_histories:
            user_code_list = [get_code(uid) for uid in code_history.user_code_history]
            code_versions = CodeHistoryView(
                user_code_history=user_code_list,
                service_func_name=code_history.service_func_name,
                comment_history=code_history.comment_history,
            )
            code_versions_dict[code_history.service_func_name] = code_versions
        return CodeHistoriesDict(code_versions=code_versions_dict)

    @service_method(
        path="code_history.get_history",
        name="get_history",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_histories_for_current_user(
        self, context: AuthedServiceContext
    ) -> CodeHistoriesDict:
        return self.fetch_histories_for_user(
            context=context, user_verify_key=context.credentials
        )

    @service_method(
        path="code_history.get_history_for_user",
        name="get_history_for_user",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_history_for_user(
        self, context: AuthedServiceContext, email: str
    ) -> CodeHistoriesDict:
        user = context.server.services.user.stash.get_by_email(
            credentials=context.credentials, email=email
        ).unwrap()
        return self.fetch_histories_for_user(
            context=context, user_verify_key=user.verify_key
        )

    @service_method(
        path="code_history.get_histories",
        name="get_histories",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_histories_group_by_user(
        self, context: AuthedServiceContext
    ) -> UsersCodeHistoriesDict:
        if context.role in [ServiceRole.DATA_OWNER, ServiceRole.ADMIN]:
            code_histories = self.stash.get_all(
                context.credentials, has_permission=True
            ).unwrap()
        else:
            code_histories = self.stash.get_all(context.credentials).unwrap()

        users = context.server.services.user.stash.get_all(context.credentials).unwrap()
        user_code_histories = UsersCodeHistoriesDict(server_uid=context.server.id)

        verify_key_2_user_email = {}
        for user in users:
            user_code_histories.user_dict[user.email] = []
            verify_key_2_user_email[user.verify_key] = user.email

        for code_history in code_histories:
            user_email = verify_key_2_user_email[code_history.user_verify_key]
            user_code_histories.user_dict[user_email].append(
                code_history.service_func_name
            )

        return user_code_histories

    @service_method(
        path="code_history.get_by_name_and_user_email",
        name="get_by_name_and_user_email",
    )
    def get_by_func_name_and_user_email(
        self,
        context: AuthedServiceContext,
        service_func_name: str,
        user_email: str,
        user_id: UID,
    ) -> list[CodeHistory]:
        user_verify_key = context.server.services.user.user_verify_key(user_email)

        filters = {
            "id": user_id,
            "email": user_email,
            "verify_key": user_verify_key,
            "service_func_name": service_func_name,
        }

        return self.stash.get_all(
            credentials=context.credentials, filters=filters
        ).unwrap()
