"""The Google Drive API client that the Drive transport jobs share.

One client per connection. It holds the Drive service, which is not safe to
use from more than one thread.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class GDriveClient:
    # A googleapiclient Drive v3 service, or a MockDriveService in tests.
    drive_service: Any
