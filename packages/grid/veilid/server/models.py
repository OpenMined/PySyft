# third party
from pydantic import BaseModel


class ResponseModel(BaseModel):
    message: str
