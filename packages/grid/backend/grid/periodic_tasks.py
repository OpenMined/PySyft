# syft absolute
from typing import Dict
from collections import defaultdict
from syft.core.common.message import SignedImmediateSyftMessageWithoutReply
from syft.core.node.common.util import get_s3_client

# grid absolute
from grid.core.celery_app import celery_app
from grid.core.node import node


@celery_app.task
def cleanup_seaweed():
    client = get_s3_client(settings=node.settings)
    incomplete_upload_objs = client.list_multipart_uploads(Bucket=node.id.no_dash)['Uploads']
    for obj in incomplete_upload_objs:
        # Abort multipart upload
        client.abort_multipart_upload(
            UploadId=obj['UploadId'],
            Key=obj['Key'],
            Bucket=node.id.no_dash,
        )


@celery_app.on_after_configure.connect
def setup_periodic_task(sender, **kwargs) -> None:  # type: ignore
    celery_app.add_periodic_task(200.0, cleanup_seaweed.s(), name='Seaweed FS ./upload CLEANUP')
