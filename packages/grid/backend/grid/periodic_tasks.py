# stdlib
import datetime

# third party
import pytz

# syft absolute
from syft.core.node.common.util import get_s3_client

# grid absolute
from grid.core.celery_app import celery_app
from grid.core.node import node


@celery_app.task
def check_tasks_to_be_executed() -> None:
    tasks = node.tasks.find(search_params={"status": "accepted"})
    for task in tasks:
        if task.execution["status"] == "enqueued":
            celery_app.send_task(
                "grid.worker.execute_task",
                args=[task.uid, task.code, task.inputs, task.outputs],
            )


@celery_app.task
def cleanup_incomplete_uploads_from_blob_store() -> bool:
    """Cleans up incomplete uploads from blob storage."""

    DAYS_TO_RETAIN = 1

    # Get current time in UTC timezone
    now = datetime.datetime.now(pytz.timezone("UTC"))

    client = get_s3_client(settings=node.settings)
    incomplete_upload_objs = client.list_multipart_uploads(Bucket=node.id.no_dash).get(
        "Uploads", []
    )

    for obj in incomplete_upload_objs:

        # Get the upload id and object name
        upload_id: str = obj["UploadId"]
        obj_name: str = obj["Key"]

        # Get the list of all parts of the object uploaded
        # This step is required to get the upload time of the object
        object_parts: list = client.list_parts(
            Bucket=node.id.no_dash, UploadId=upload_id, Key=obj_name
        ).get("Parts", [])

        obj_part_expired = False
        for part in object_parts:
            # Normalize upload time to UTC timezone
            part_upload_time = pytz.timezone("UTC").normalize(part["LastModified"])

            # If upload time of any part of the object
            # crosses DAYS_TO_RETAIN, then expire the whole object
            if (now - part_upload_time).days > DAYS_TO_RETAIN:
                obj_part_expired = True
                break

        if obj_part_expired:
            # Abort multipart upload
            client.abort_multipart_upload(
                UploadId=upload_id,
                Key=obj_name,
                Bucket=node.id.no_dash,
            )

    return True
