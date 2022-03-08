# stdlib
from typing import Any

# third party
from fastapi import APIRouter
from fastapi import Depends
from fastapi.responses import JSONResponse

# syft absolute
from syft.core.node.common.util import get_s3_client

# grid absolute
from grid.api.dependencies.current_user import get_current_user
from grid.core.node import node

router = APIRouter()


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
    download_url = (
        s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": node.name, "Key": key},
            ExpiresIn=1800,  # expiration time of the url in seconds
            HttpMethod="GET",
        )
        if s3_client
        else ""
    )

    response = {"url": download_url}
    return response
