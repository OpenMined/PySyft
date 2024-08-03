# stdlib

# third party
from pydantic import BaseModel, ConfigDict


class SyftBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
