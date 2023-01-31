# stdlib
from typing import Dict
from typing import Optional
from typing import Union

# third party
from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from loguru import logger
from pydantic import BaseModel
from starlette import status

# syft absolute
from syft.core.node.common.node_service.generic_payload.syft_message import (
    NewSyftMessage,
)
from syft.core.node.common.node_service.task_submission.task_submission import (
    CreateTask as CreateTaskMessage,
)
from syft.core.node.common.node_service.task_submission.task_submission import (
    GetTask as GetTaskMessage,
)
from syft.core.node.common.node_service.task_submission.task_submission import (
    GetTasks as GetTasksMessage,
)
from syft.core.node.common.node_service.task_submission.task_submission import (
    ReviewTask,
)

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.api.users.models import UserPrivate
from grid.core.node import node

# relative
from .models import CreateTaskModel
from .models import GetTasks
from .models import ReviewTaskModel
from .models import StdResponseMessage
from .models import Task
from .models import TaskErrorResponse

router = APIRouter()


def process_task_requests(
    user: UserPrivate,
    msg_class: NewSyftMessage,
    return_type: BaseModel,
    request: Optional[Dict[str, str]] = None,
) -> BaseModel:
    if request is None:
        request = {}

    msg = msg_class(address=node.address, reply_to=node.address, kwargs=request).sign(
        signing_key=user.get_signing_key()
    )

    reply = node.recv_immediate_msg_with_reply(msg)
    return return_type(**reply.message.kwargs)


@router.post(
    "",
    response_model=StdResponseMessage,
    name="tasks:create",
    status_code=status.HTTP_200_OK,
)
async def create_task(
    task: CreateTaskModel = Body(...), user: UserPrivate = Depends(get_current_user)
) -> StdResponseMessage:
    try:
        return process_task_requests(
            user, CreateTaskMessage, StdResponseMessage, request=task.dict()
        )
    except Exception as err:
        logger.error(err)
        return TaskErrorResponse(error=str(err))


@router.get(
    "/{task_uid}",
    response_model=Task,
    name="tasks:read",
    status_code=status.HTTP_200_OK,
)
async def get_task(
    task_uid: str, user: UserPrivate = Depends(get_current_user)
) -> Task:
    try:
        return process_task_requests(
            user, GetTaskMessage, Task, request={"task_uid": task_uid}
        )
    except Exception as err:
        logger.error(err)
        return TaskErrorResponse(error=str(err))


@router.get(
    "", response_model=GetTasks, name="tasks:read_all", status_code=status.HTTP_200_OK
)
async def get_tasks(user: UserPrivate = Depends(get_current_user)) -> GetTasks:
    try:
        return process_task_requests(user, GetTasksMessage, GetTasks)
    except Exception as err:
        logger.error(err)
        return TaskErrorResponse(error=str(err))


@router.put(
    "/{task_uid}",
    response_model=StdResponseMessage,
    name="tasks:review",
    status_code=status.HTTP_200_OK,
)
async def review_task(
    task_uid: str,
    review_task: ReviewTaskModel = Body(...),
    user: UserPrivate = Depends(get_current_user),
) -> Union[StdResponseMessage, Dict[str, str]]:
    try:
        review_task.task_uid = task_uid
        return process_task_requests(
            user, ReviewTask, StdResponseMessage, request=review_task
        )
    except Exception as err:
        logger.error(err)
        return TaskErrorResponse(error=str(err))
