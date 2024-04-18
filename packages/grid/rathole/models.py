# third party
from pydantic import BaseModel


class ResponseModel(BaseModel):
    message: str


class RatholeConfig(BaseModel):
    uuid: str
    secret_token: str
