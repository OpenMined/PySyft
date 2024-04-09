# third party
from pydantic import BaseModel


class ResponseModel(BaseModel):
    message: str


class CPUAttestationResponseModel(BaseModel):
    result: str
    vendor: str | None = None  # Hardware Manufacturer


class GPUAttestationResponseModel(BaseModel):
    result: str
    vendor: str | None = None  # Hardware Manufacturer
