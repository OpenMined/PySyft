# third party
from pydantic import BaseModel


class SyftBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
