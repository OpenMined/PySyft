# stdlib
from datetime import date
from io import StringIO
import random
import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

# third party
import requests
from result import Err
from result import Ok
from result import Result

# relative
from .....logger import logger
from .....oblv.constants import LOCAL_MODE
from .....oblv.constants import LOCAL_MODE_CONNECTION_PORT
from ....common.serde.deserialize import _deserialize as deserialize
from ....common.serde.serializable import serializable
from ....common.uid import UID
from ..action_object import ActionObject
from ..action_store import ActionStore
from ..api import SyftAPI
from ..context import AuthedServiceContext
from ..document_store import DocumentStore
from ..response import SyftError
from ..response import SyftSuccess
from ..service import AbstractService
from ..service import service_method
from .oblv_keys_stash import OblvKeysStash
from .oblv_service import OBLV_PROCESS_CACHE
from .oblv_service import make_request_to_enclave
from .task import DictObject
from .task import NodeView
from .task import Task
from .task_stash import TaskStash
from .util import find_available_port

stdout_ = sys.stdout
stderr_ = sys.stderr


@serializable(recursive_serde=True)
class TaskService(AbstractService):
    document_store: DocumentStore
    task_stash: TaskStash
    oblv_keys_stash: OblvKeysStash
    action_store: ActionStore

    def __init__(
        self, document_store: DocumentStore, action_store: ActionStore
    ) -> None:
        self.document_store = document_store
        self.task_stash = TaskStash(store=document_store)
        self.oblv_keys_stash = OblvKeysStash(store=document_store)
        self.action_store = action_store

    @service_method(path="task.create_enclave_task", name="create_enclave_task")
    def create_enclave_task(
        self,
        context: AuthedServiceContext,
        inputs: Dict[NodeView, dict],
        code: str,
        outputs: List[str],
        owners: List[NodeView],
        task_id: UID,
        oblv_metadata: Optional[dict] = None,
    ) -> Result[Ok, Err]:
        """Enclave Submit task"""
        # TODO 🟣 Check for permission after it is fully integrated

        if not owners:
            return Err("Enclave task creation should contain atleast one owner")
        task = Task(
            id=task_id,
            inputs=inputs,
            code=code,
            outputs={var: " -- " for var in outputs},
            user=str(context.credentials),
            owners=owners,
            status={owner: "pending" for owner in owners},
            created_at=date.today().strftime("%d/%m/%Y %H:%M:%S"),
            updated_at=" -- ",
            reviewed_by=" -- ",
            execution="pending",
            oblv_metadata=oblv_metadata,
        )

        result = self.task_stash.set(task)

        if result.is_ok():
            return Ok(SyftSuccess(message=f"Added task to enclave with id: {task_id}"))

        return result.err()

    @service_method(path="task.create_domain_task", name="create_domain_task")
    def create_domain_task(
        self,
        context: AuthedServiceContext,
        inputs: Dict[NodeView, dict],
        code: str,
        outputs: List[str],
        owners: List[NodeView],
        task_id: UID,
        oblv_metadata: Optional[dict] = None,
    ) -> Result[Ok, Err]:
        """Domain Submit task"""
        # TODO 🟣 Check for permission after it is fully integrated

        if owners and len(owners) != 1:
            return Err("Domain task creation should have exactly one owner")
        task = Task(
            id=task_id,
            inputs=inputs,
            code=code,
            outputs={var: " -- " for var in outputs},
            user=str(context.credentials),
            owners=owners,
            status={owners[0]: "pending"},
            created_at=date.today().strftime("%d/%m/%Y %H:%M:%S"),
            updated_at=" -- ",
            reviewed_by=" -- ",
            execution="pending",
            oblv_metadata=oblv_metadata,
        )

        result = self.task_stash.set(task)

        if result.is_ok():
            return Ok(
                SyftSuccess(message=f"Added task to domain node: {owners[0].name}")
            )

        return result.err()

    @service_method(path="task.get_task", name="get_task")
    def get_task(
        self,
        context: AuthedServiceContext,
        task_id: UID,
    ) -> Result[Ok, Err]:
        # TODO 🟣 Check for permission after it is fully integrated

        res = self.task_stash.get_by_uid(task_id)
        if res.is_ok():
            return res.ok()
        return res.err()

    @service_method(path="task.review_task", name="review_task")
    def review_task(
        self,
        context: AuthedServiceContext,
        task_id: UID,
        approve: bool,
        reason: str = "",
    ) -> Result[Ok, Err]:
        # TODO 🟣 Check for permission after it is fully integrated

        task = self.task_stash.get_by_uid(task_id)
        if task.is_ok():
            # 🟡 TODO: To Remove double nesting of result variable in Collection access
            task = task.ok().ok()
        else:
            return task.err()

        owner = task.owners[0]
        if task.status[owner] in ["Approved", "Denied"]:
            return Err(
                SyftError(
                    message=f"Cannot Modify the status of task: {task.id} which has been Approved/Denied."
                    + "Kindly Submit a new request"
                )
            )

        task.reason = reason
        task.updated_at = date.today().strftime("%d/%m/%Y %H:%M:%S")

        # Fetch private data from action store if the code task is approved
        if approve:
            task.status[owner] = "Approved"

            # Retrieving input map of the current domain
            owner_input_map = task.inputs[owner]
            private_input_map = {}

            # Replace inputs with private data
            for obj_name, obj_id in owner_input_map.items():
                result = self.action_store.get(
                    uid=obj_id, credentials=context.credentials
                )
                if result.is_ok():
                    private_input_map[obj_name] = result.ok()
                else:
                    return Err(result.err())

        else:
            task.status[task.owners[0]] = "Denied"

        # Update task status back to DB
        res = self.task_stash.update(task)

        # If we are in the Enclave execution and have metadata for enclaves
        # Sent the task status to the connected enclave
        if task.oblv_metadata:
            api = self._get_api(task.oblv_metadata)
            enclave_res = api.services.task.send_status_to_enclave(
                task, private_input_map
            )
            if isinstance(enclave_res, bool) and enclave_res:
                return Ok(
                    SyftSuccess(message=f"Sent task: {task_id} status to enclave")
                )
            return Err(f"{enclave_res}")

        if res.is_ok():
            return Ok(SyftSuccess(message=f"Task: {task_id}  - {approve}"))
        return res.err()

    def _get_api(self, oblv_metadata: dict) -> SyftAPI:
        deployment_id = oblv_metadata["deployment_id"]
        oblv_client = oblv_metadata["oblv_client"]
        if not LOCAL_MODE:
            if (
                deployment_id in OBLV_PROCESS_CACHE
                and OBLV_PROCESS_CACHE[deployment_id][0].poll() is None
            ):
                port = OBLV_PROCESS_CACHE[deployment_id][1]
            else:
                # randomized port staring point, to quickly find free port
                port_start = 3000 + random.randint(1, 10_000)
                port = find_available_port(
                    host="127.0.0.1", port=port_start, search=True
                )
            connection_string = f"http://127.0.0.1:{port}"
        else:
            port = LOCAL_MODE_CONNECTION_PORT
            connection_string = (
                f"http://host.docker.internal:{LOCAL_MODE_CONNECTION_PORT}"
            )
        req = make_request_to_enclave(
            connection_string=connection_string + "/worker/api",
            deployment_id=deployment_id,
            oblv_client=oblv_client,
            oblv_keys_stash=self.oblv_keys_stash,
            request_method=requests.get,
            connection_port=port,
        )

        obj = deserialize(req.content, from_bytes=True)
        obj.api_url = f"{connection_string}/worker/syft_api_call"
        return cast(SyftAPI, obj)

    @service_method(path="task.send_status_to_enclave", name="send_status_to_enclave")
    def send_status_to_enclave(
        self, context: AuthedServiceContext, task: Task, private_input_map: dict
    ) -> Result[Ok, Err]:
        enclave_task = self.task_stash.get_by_uid(task.id)

        if enclave_task.is_ok():
            # 🟡 TODO: To Remove double nesting of result variable in Collection access
            enclave_task = enclave_task.ok().ok()
        else:
            return enclave_task.err()

        task_owner = task.owners[0]
        if task_owner in enclave_task.owners:
            if enclave_task.status[task_owner] in ["Approved", "Denied"]:
                return Err(
                    SyftSuccess(
                        message=f"Cannot Modify the status of task: {task.id} which has been Approved/Denied \n"
                        + "Kindly Submit a new request"
                    )
                )
            enclave_task.status[task_owner] = task.status[task_owner]
            # Create nested action store of DO for storing intermediate data
            if not self.action_store.exists(task_owner.node_uid):
                dict_object = DictObject()
                dict_object.base_dict[task.id] = private_input_map
                self.action_store.set(
                    uid=task_owner.node_uid,
                    credentials=context.credentials,
                    syft_object=dict_object,
                )
            else:
                result = self.action_store.get(
                    uid=task_owner.node_uid, credentials=context.credentials
                )
                if result.is_ok():
                    result = result.ok()
                    result.base_dict[task.id] = private_input_map
                    self.action_store.set(
                        uid=task_owner.node_uid,
                        credentials=context.credentials,
                        syft_object=result,
                    )
                else:
                    return result.err()

            # Check if all the DO approved the code execution
            code_status_check = set(enclave_task.status.values())
            if len(code_status_check) == 1 and "Approved" in code_status_check:
                enclave_task.execution = "executing"
                self.task_stash.update(enclave_task)
                # TODO 🟣 Branch to separate thread for execution of enclave task
                self.execute_task(enclave_task=enclave_task, context=context)

            self.task_stash.update(enclave_task)

            return Ok(True)
        else:
            return Err(
                f"Task: {task.id.no_dash}:{enclave_task.owners} in enclave does not contain {task.owners[0].name} "
            )

    def fetch_private_inputs(
        self, enclave_task: Task, context: AuthedServiceContext
    ) -> Dict:
        inputs = {}
        for domain in enclave_task.owners:
            domain_input = self.action_store.get(
                uid=domain.node_uid,
                credentials=context.credentials,
                skip_permission=True,
            )
            if domain_input.is_ok():
                inputs.update(domain_input.ok().base_dict[enclave_task.id])
            else:
                return domain_input.err()
        return inputs

    def execute_task(self, enclave_task: Task, context: AuthedServiceContext) -> None:
        global stdout_
        global stderr_

        code = enclave_task.code
        inputs = self.fetch_private_inputs(enclave_task=enclave_task, context=context)
        outputs = enclave_task.outputs

        try:
            logger.info(f"inital outputs: {outputs}")

            # Check overlap between inputs and vars
            global_input_inter = set(globals().keys()).intersection(set(inputs.keys()))
            local_input_inter = set(vars().keys()).intersection(set(inputs.keys()))

            # If there's some intersection between global variables and input
            if global_input_inter or local_input_inter:
                stderr_message = " You can't use variable name: "
                stderr_message += ",".join(list(global_input_inter))
                stderr_message += ",".join(list(local_input_inter))

                enclave_task.execution = "failed"
                self.task_stash.update(enclave_task)
                return Err("Variable conflicts in global space")

            # create file-like string to capture ouputs
            codeOut = StringIO()
            codeErr = StringIO()

            sys.stdout = codeOut
            sys.stderr = codeErr

            locals().update(inputs)
            # byte_code = compile_restricted(code, "<string>", "exec")
            # exec(byte_code, restricted_globals)
            exec(code, locals(), locals()) in {}  # nosec

            for output in outputs:
                logger.info(f"variable: {output} result: {vars()[output]}")
                outputs[output] = vars()[output]

            # restore stdout and stderr
            sys.stdout = stdout_
            sys.stderr = stderr_

            logger.info(outputs)

            logger.info("Error: " + str(codeErr.getvalue()))
            logger.info("Std ouputs: " + str(codeOut.getvalue()))

            new_id = UID()

            dict_object = DictObject()
            dict_object.base_dict = outputs
            print("dict object", dict_object)
            self.action_store.set(
                uid=new_id,
                credentials=context.credentials,
                syft_object=dict_object,
            )
            print("Output id", new_id)
            enclave_task.outputs = {"output_id": new_id}
            enclave_task.execution = "Done"
            self.task_stash.update(enclave_task)
        except Exception as e:
            sys.stdout = stdout_
            sys.stderr = stderr_
            raise e
            print("Task Failed with Exception", e)
        finally:
            sys.stdout = stdout_
            sys.stderr = stderr_

    @service_method(path="task.get", name="get")
    def get(self, context: AuthedServiceContext, uid: UID) -> Result[ActionObject, str]:
        """Get an object from the action store"""
        # TODO 🟣 Check for permission after it is fully integrated
        # The method should also be removed when we have permissions integrated
        # as we could access the action_store by the action service
        result = self.action_store.get(
            uid=uid, credentials=context.credentials, skip_permission=True
        )
        if result.is_ok():
            return Ok(result.ok())
        return Err(result.err())

    # Below are convenience methods to be used for testing code execution requests
    # Below method checks if  the domain nodes are able to connect to enclaves
    @service_method(path="task.send_hello_to_enclave", name="send_hello_to_enclave")
    def send_hello_to_enclave(
        self, context: AuthedServiceContext, oblv_metadata: dict
    ) -> Result[Ok, Err]:
        api = self._get_api(oblv_metadata)
        res = api.services.test.send_name("Natsu")
        return Ok(res)
