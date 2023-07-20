# stdlib
from typing import List
from typing import Union

# third party
from result import Err
from result import Ok
from result import Result

# relative
from ...store.document_store import DocumentStore
from ...types.uid import UID
from ..code.user_code import SubmitUserCode
from ..code.user_code import UserCode
from ..context import AuthedServiceContext
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from ..user.user import User
from ..user.user_roles import DATA_SCIENTIST_ROLE_LEVEL
from .code_history import CodeHistory
from .code_history_stash import CodeHistoryStash


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
        self, context: AuthedServiceContext, code: Union[SubmitUserCode, UserCode]
    ) -> Union[SyftSuccess, SyftError]:
        user_code_service = context.node.get_service("usercodeservice")

        if isinstance(code, SubmitUserCode):
            result = user_code_service.submit(context=context, code=code)
            if isinstance(result, SyftError):
                return result

            uid = UID.from_string(result.message.split(" ")[-1])
            code = user_code_service.get_by_uid(context=context, uid=uid)

        elif isinstance(code, UserCode):
            result = user_code_service.get_by_uid(context=context, uid=code.id)
            if isinstance(result, SyftError):
                return result
            code = result

        result_code_history_list = self.stash.get_by_service_func_name(
            credentials=context.credentials, service_func_name=code.service_func_name
        )

        if result_code_history_list.is_err():
            return SyftError(message=result_code_history_list.err())

        code_history = None
        code_history_list = result_code_history_list.ok()

        for elem in code_history_list:
            if elem.user_verify_key == context.credentials:
                code_history = elem

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

        code_history.add_code(code=code)
        result = self.stash.update(credentials=context.credentials, obj=code_history)
        if result.is_err():
            return SyftError(message=result.err())

        return SyftSuccess(message="Code version submit success")

    @service_method(
        path="code_history.get_all", name="get_all", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_all(
        self, context: AuthedServiceContext
    ) -> Union[List[UserCode], SyftError]:
        """Get a Dataset"""
        result = self.stash.get_all(context.credentials)
        if result.is_ok():
            return result.ok()
        return SyftError(message=result.err())

    @service_method(
        path="code_history.get_by_id", name="get_by_id", roles=DATA_SCIENTIST_ROLE_LEVEL
    )
    def get_code_by_uid(
        self, context: AuthedServiceContext, uid: UID
    ) -> Union[SyftSuccess, SyftError]:
        """Get a User Code Item"""
        result = self.stash.get_by_uid(context.credentials, uid=uid)
        if result.is_ok():
            code_history = result.ok()
            return code_history
        return SyftError(message=result.err())

    @service_method(path="code_history.delete_by_id", name="delete_by_id")
    def delete(self, context: AuthedServiceContext, uid: UID):
        result = self.stash.delete_by_uid(context.credentials, uid)
        if result.is_ok():
            return result.ok()
        else:
            return SyftError(message=result.err())

    # @service_method(path="code_history.get_by_name_and_user_id", name="get_by_name_and_user_id")
    # def get_by_name_and_user_id(self, context: AuthedServiceContext, service_func_name: str, user_id: UID
    # ) -> Union[SyftSuccess, SyftError]:

    #     user = self.verify_user_id(context=context, user_id=user_id)
    #     print("USER: ", user.value)
    #     if user.err():
    #         return user

    #     kwargs = {
    #         "id": user_id,
    #         "verify_key": user.value.verify_key,
    #         "service_func_name": service_func_name
    #         }

    #     #    kwargs = user_search.to_dict(exclude_empty=True)

    #     # UserExperience
    #     result = self.stash.find_all(credentials=context.credentials, **kwargs)
    #     print("Results", result)
    #     if result.is_err(): #or len(result) > 1
    #         return result

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
    ) -> Union[SyftSuccess, SyftError]:
        user_service = context.node.get_service("userservice")
        user_verify_key = user_service.user_verify_key(user_email)

        # user = self.verify_user_id(context=context, user_id=user_id)
        print("USER: ", user_verify_key)
        if isinstance(user_verify_key, SyftError):
            return user_verify_key

        kwargs = {
            "id": user_id,
            "email": user_email,
            "verify_key": user_verify_key,
            "service_func_name": service_func_name,
        }

        result = self.stash.find_all(credentials=context.credentials, **kwargs)
        print("Results", result)
        if result.is_err():  # or len(result) > 1
            return result

    # @service_method(path="code_history.get_by_name", name="get_by_name", roles=DATA_SCIENTIST_ROLE_LEVEL)
    # def get_by_func_name(self, context: AuthedServiceContext, service_func_name: str, user_email: str):
    #     user_service = context.node.get_service("userservice")
    #     user_verify_key = user_service.user_verify_key(user_email)

    # #     def get_by_name(
    # #     self, credentials: SyftVerifyKey, name: str
    # # ) -> Result[Optional[Dataset], str]:
    #     qks = QueryKeys(qks=[NamePartitionKey.with_obj(name)])
    #     return self.query_one(credentials=credentials, qks=qks)

    # TODO
    # verify the user => get_service from user_service
    # create kwargs => get syft_Verify key from the user
    # try find_all
    # make sure we get atmost one result => error
    # Write another function for data scientist => get_by_name
    # Another function to check for the syft_verify key
    #
