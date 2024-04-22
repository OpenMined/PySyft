# third party
from pydantic import BaseModel


class ResponseModel(BaseModel):
    message: str


class RatholeConfig(BaseModel):
    uuid: str
    secret_token: str
    local_addr_host: str
    local_addr_port: int

    @property
    def local_address(self) -> str:
        return f"{self.local_addr_host}:{self.local_addr_port}"
