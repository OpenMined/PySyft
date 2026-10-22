from pathlib import Path
from typing import Iterator

from ..job_ref import RequestRef
from ..submission import REQUEST_FILE_NAME
from .v1 import ProtocolCodecV1


class ProtocolCodecV2(ProtocolCodecV1):
    """The v1 layout under <ds_email>/v2/, plus api calls in requests/."""

    version = "2"
    protocol_versions = ["2"]

    def request_dir(self, ref: RequestRef) -> Path:
        return self.config.get_request_job_dir(
            ref.datasite_email, ref.ds_email, ref.job_name, ref.protocol_version
        )

    def iter_request_refs(self, datasite_email: str) -> Iterator[RequestRef]:
        root = self.config.get_requests_dir(datasite_email)
        yield from self._scan(
            root, datasite_email, REQUEST_FILE_NAME, ref_type=RequestRef
        )
