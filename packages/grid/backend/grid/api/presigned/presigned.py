# stdlib
import math
from typing import Any
from typing import Optional

# third party
from fastapi import APIRouter
from fastapi import Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.config import settings
from grid.core.node import node
from grid.core.s3 import get_s3_client

router = APIRouter()


class FileUpload(BaseModel):
    filename: str
    file_size: int
    chunk_size: int
    content_type: Optional[str] = None


@router.post("/upload", status_code=201, response_class=JSONResponse)
def presigned_upload(
    body: FileUpload,
    current_user: Any = Depends(get_current_user),
) -> Any:

    user_role = node.roles.first(**{"id": current_user.role})
    if not user_role.can_upload_data:
        return {"message": "You're not authorized to do this."}

    key = f"{body.filename}"
    s3_client = get_s3_client(docker_host=True)
    result = s3_client.create_multipart_upload(Bucket=settings.S3_BUCKET, Key=key)
    total_parts = math.ceil(body.file_size / body.chunk_size)
    upload_id = result["UploadId"]
    signed_urls = []
    s3_client = get_s3_client()
    for part_no in range(1, total_parts + 1):
        # Creating presigned urls
        signed_url = s3_client.generate_presigned_url(
            ClientMethod="upload_part",
            Params={
                "Bucket": settings.S3_BUCKET,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": part_no,
            },
            ExpiresIn=1800,  # in seconds
        )
        signed_urls.append({"part_no": part_no, "url": signed_url})

    response = {"upload_id": upload_id, "parts": signed_urls}
    return response


class FileUploadComplete(BaseModel):
    filename: str
    upload_id: str
    parts: list


@router.post("/upload/complete", status_code=204, response_class=JSONResponse)
def presigned_upload_complete(
    body: FileUploadComplete,
    current_user: Any = Depends(get_current_user),
) -> Any:

    user_role = node.roles.first(**{"id": current_user.role})
    if not user_role.can_upload_data:
        return {"message": "You're not authorized to do this."}

    key = f"{body.filename}"
    s3_client = get_s3_client(True)
    result = s3_client.complete_multipart_upload(
        Bucket=settings.S3_BUCKET,
        Key=key,
        MultipartUpload={"Parts": body.parts},
        UploadId=body.upload_id,
    )

    response = {"message": "Upload completed.", "result": result}
    return response


class DownloadQueryParams:
    def __init__(self, filename: str) -> None:
        self.filename = filename


@router.get("/download", status_code=200, response_class=JSONResponse)
def presigned_download(
    query_params: DownloadQueryParams = Depends(),
    current_user: Any = Depends(get_current_user),
) -> Any:

    # TODO: Need to update this permission.
    user_role = node.roles.first(**{"id": current_user.role})
    if not user_role.can_upload_data:
        return {"message": "You're not authorized to do this."}

    s3_client = get_s3_client()

    key = f"{query_params.filename}"
    download_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET, "Key": key},
        ExpiresIn=1800,  # expiration time of the url in seconds
        HttpMethod="GET",
    )

    response = {"url": download_url}
    return response
