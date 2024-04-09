# stdlib

# relative
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ...util.telemetry import instrument
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user_roles import DATA_OWNER_ROLE_LEVEL
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .code_history import CodeHistoriesDict
from .code_history import CodeHistory
from .code_history import CodeHistoryView
from .code_history import UsersCodeHistoriesDict
from .code_history_stash import CodeHistoryStash


@instrument
@serializable()
class CodeHistoryService(AbstractService):
    store: DocumentStore
    stash: CodeHistoryStash

    def __init__(self, store: DocumentStore) -> None:
        self.store = store
        self.stash = CodeHistoryStash(store=store)

    @service_method(
        path="code_history.submit_version",
        name="submit_version",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def submit_version(
        self,
        context: AuthedServiceContext,
        code: SubmitUserCode | UserCode,
        comment: str | None = None,
    ) -> SyftSuccess | SyftError:
        user_code_service = context.node.get_service("usercodeservice")
        if isinstance(code, SubmitUserCode):
            result = user_code_service._submit(context=context, code=code)
            if result.is_err():
                return SyftError(message=str(result.err()))
            code = result.ok()
        elif isinstance(code, UserCode):  # type: ignore[unreachable]
            result = user_code_service.get_by_uid(context=context, uid=code.id)
            if isinstance(result, SyftError):
                return result
            code = result

        result = self.stash.get_by_service_func_name_and_verify_key(
            credentials=context.credentials,
            service_func_name=code.service_func_name,
            user_verify_key=context.credentials,
        )

        if result.is_err():
            return SyftError(message=result.err())

        code_history: CodeHistory | None = result.ok()

        if code_history is None:
            code_history = CodeHistory(
                id=UID(),
                node_uid=context.node.id,
                user_verify_key=context.credentials,
                service_func_name=code.service_func_name,
            )
            result = self.stash.set(credentials=context.credentials, obj=code_history)
            if result.is_err():
                return SyftError(message=result.err())

        code_history.add_code(code=code, comment=comment)
        result = self.stash.update(credentials=context.credentials, obj=code_history)
        if result.is_err():
            return SyftError(message=result.err())

        return SyftSuccess(message="Code version submit success")

    @service_method(
        path="code_history.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_all(self, context: AuthedServiceContext) -> list[CodeHistory] | SyftError:
        """Get a Dataset"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(
        path="code_history.get", name="get", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_code_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        """Get a User Code Item"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            code_history = result.ok()
            return code_history
        return SyftError(message=result.err())

    @service_method(path="code_history.delete", name="delete")
    def delete(
        self, context: AuthedServiceContext, uid: UID
    ) -> SyftSuccess | SyftError:
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_ok():
            return result.ok()
        else:
            return SyftError(message=result.err())

    def fetch_histories_for_user(
        self, context: AuthedServiceContext, user_verify_key: SyftVerifyKey
    ) -> CodeHistoriesDict | SyftError:
        result = self.stash.get_by_verify_key(
            credentials=context.credentials, user_verify_key=user_verify_key
        )

        user_code_service = context.node.get_service("usercodeservice")

        def get_code(uid: UID) -> UserCode | SyftError:
            return user_code_service.get_by_uid(context=context, uid=uid)

        if result.is_ok():
            code_histories = result.ok()
            code_versions_dict = {}

            for code_history in code_histories:
                user_code_list = []
                for uid in code_history.user_code_history:
                    user_code_list.append(get_code(uid))
                code_versions = CodeHistoryView(
                    user_code_history=user_code_list,
                    service_func_name=code_history.service_func_name,
                    comment_history=code_history.comment_history,
                )
                code_versions_dict[code_history.service_func_name] = code_versions
            return CodeHistoriesDict(code_versions=code_versions_dict)
        else:
            return SyftError(message=result.err())

    @service_method(
        path="code_history.get_history",
        name="get_history",
        roles=DATA_SCIENTIST_ROLE_LEVEL,
    )
    def get_histories_for_current_user(
        self, context: AuthedServiceContext
    ) -> CodeHistoriesDict | SyftError:
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
    ) -> CodeHistoriesDict | SyftError:
        user_service = context.node.get_service("userservice")
        result = user_service.stash.get_by_email(
            credentials=context.credentials, email=email
        )
        if result.is_ok():
            user = result.ok()
            return self.fetch_histories_for_user(
                context=context, user_verify_key=user.verify_key
            )
        return SyftError(message=result.err())

    @service_method(
        path="code_history.get_histories",
        name="get_histories",
        roles=DATA_OWNER_ROLE_LEVEL,
    )
    def get_histories_group_by_user(
        self, context: AuthedServiceContext
    ) -> UsersCodeHistoriesDict | SyftError:
        result = self.stash.get_all(credentials=context.credentials)
        if result.is_err():
            return SyftError(message=result.err())
        code_histories: list[CodeHistory] = result.ok()

        user_service = context.node.get_service("userservice")
        result = user_service.stash.get_all(context.credentials)
        if result.is_err():
            return SyftError(message=result.err())
        users = result.ok()

        user_code_histories = UsersCodeHistoriesDict(node_uid=context.node.id)

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
    ) -> list[CodeHistory] | SyftError:
        user_service = context.node.get_service("userservice")
        user_verify_key = user_service.user_verify_key(user_email)

        if isinstance(user_verify_key, SyftError):
            return user_verify_key

        kwargs = {
            "id": user_id,
            "email": user_email,
            "verify_key": user_verify_key,
            "service_func_name": service_func_name,
        }

        result = self.stash.find_all(credentials=context.credentials, **kwargs)
        if result.is_err():  # or len(result) > 1
            return result

        return result.ok()
