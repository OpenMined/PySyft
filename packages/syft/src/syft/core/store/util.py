# stdlib
from typing import Dict
from typing import Optional

# third party
import boto3
from botocore.awsrequest import prepare_request_dict
from botocore.client import Config
from botocore.signers import _should_use_global_endpoint
from pydantic import BaseSettings

# relative
from ...grid.grid_url import GridURL


def get_s3_client(settings: BaseSettings = BaseSettings()) -> "boto3.client.S3":
    try:
        s3_endpoint = settings.S3_ENDPOINT
        s3_port = settings.S3_PORT
        s3_grid_url = GridURL(host_or_ip=s3_endpoint, port=s3_port)
        return boto3.client(
            "s3",
            endpoint_url=s3_grid_url.url,
            aws_access_key_id=settings.S3_ROOT_USER,
            aws_secret_access_key=settings.S3_ROOT_PWD,
            config=Config(signature_version="s3v4"),
            region_name=settings.S3_REGION,
        )
    except Exception as e:
        print(f"Failed to create S3 Client with {s3_endpoint} {s3_port} {s3_grid_url}")
        raise e


def custom_presigned_url(
    client: "boto3.client.S3",
    endpoint_url: str,
    ClientMethod: str,
    Params: Optional[Dict] = None,
    ExpiresIn: int = 3600,
    HttpMethod: Optional[str] = None,
) -> str:
    client_method = ClientMethod
    params = Params
    if params is None:
        params = {}
    expires_in = ExpiresIn
    http_method = HttpMethod
    context = {
        "is_presign_request": True,
        "use_global_endpoint": _should_use_global_endpoint(client),
    }
    request_signer = client._request_signer
    serializer = client._serializer
    try:
        operation_name = client._PY_TO_OP_NAME[client_method]
    except KeyError as e:
        raise e
    operation_model = client.meta.service_model.operation_model(operation_name)
    params = client._emit_api_params(params, operation_model, context)
    request_dict = serializer.serialize_to_request(params, operation_model)
    if http_method is not None:
        request_dict["method"] = http_method
    prepare_request_dict(request_dict, endpoint_url=endpoint_url, context=context)
    return request_signer.generate_presigned_url(
        request_dict=request_dict, expires_in=expires_in, operation_name=operation_name
    )
