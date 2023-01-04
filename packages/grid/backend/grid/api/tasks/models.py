# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from pydantic import BaseModel


class CreateTaskModel(BaseModel):
    code: str
    load_vars: Dict[str, str] = {}
    save_vars: List[str] = []


class ReviewTaskModel(BaseModel):
    status: str
    reason: str
    task_uid: Optional[str] = ""


class Task(BaseModel):
    code: str
    uid: str
    owner: Dict[str, str]
    created_at: str
    updated_at: str
    reviewed_by: str
    status: str
    execution: Dict[str, str]
    reason: Optional[str]
    load_vars: Dict[str, str] = {}
    saved_vars: Dict[str, str] = {}


class GetTasks(BaseModel):
    tasks: List[Task]


class TaskErrorResponse(BaseModel):
    error: str


class GetTask(BaseModel):
    code: str
    code_id: str
    created_at: str
    updated_at: str
    reviewed_by: str
    status: str
    execution: Dict[str, str]


class StdResponseMessage(BaseModel):
    message: str
