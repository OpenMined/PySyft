# third party
from pydantic import BaseModel


class ResponseModel(BaseModel):
    message: str


class TestVeilidStreamerRequest(BaseModel):
    expected_response_length: int
    random_padding: str


class TestVeilidStreamerResponse(BaseModel):
    received_request_body_length: int
    random_padding: str
