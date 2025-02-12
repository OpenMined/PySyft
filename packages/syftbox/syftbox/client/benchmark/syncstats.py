import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin
from uuid import uuid4

import requests

from syftbox.client.benchmark import Stats


def generate_byte_string(size_mb: int) -> bytes:
    """Generate a sample byte string of the specified size"""
    return b"\0" * int(1024 * 1024 * size_mb)


def random_filename(size_mb: int) -> str:
    """Generate a random filename based on the size"""
    return f"{size_mb}mb-{uuid4().hex[:8]}.bytes"


@dataclass
class DataTransferStats:
    """Data transfer statistics for a specific file size"""

    file_size_mb: int
    """Size of the file in MB"""
    upload: Stats
    """Time taken to upload the file"""
    download: Stats
    """Time taken to download the file"""
    successful_runs: int
    """Number of successful runs"""
    total_runs: int
    """Total number of attempted runs"""


@dataclass
class FileTransferDuration:
    """Time taken to transfer a file"""

    upload: float
    """ Time taken to upload the file in milliseconds"""
    download: float
    """ Time taken to download the file in milliseconds"""


class SyncDataTransferStats:
    """Measure the data transfer performance of sync operations"""

    def __init__(self, url: str, token: str, email: str):
        """Initialize the server URL, token, and email"""
        self.url = url
        self.token = token
        self.email = email

    def __make_request(
        self,
        path: str,
        ignore_errors: bool = False,
        **kwargs: Any,
    ) -> float:
        """Make a request to the server and measure the time taken"""
        headers = {"Authorization": f"Bearer {self.token}", "email": self.email}
        start_time = time.time()
        url = str(urljoin(self.url, path))
        response = requests.post(
            url=url,
            headers=headers,
            **kwargs,
        )
        if not ignore_errors:
            response.raise_for_status()
        return (time.time() - start_time) * 1000

    def upload_file(self, filepath: str, data: bytes) -> float:
        """Upload a file to the server and measure the time taken"""
        files = {"file": (filepath, data, "plain/text")}
        return self.__make_request(
            "/sync/create/",
            files=files,
        )

    def download_file(self, filepath: str) -> float:
        """Download a file from the server and measure the time taken"""
        return self.__make_request("/sync/download/", json={"path": filepath})

    def delete_file(self, filepath: str) -> float:
        """Delete a file from the server and measure the time taken"""
        return self.__make_request("/sync/delete/", json={"path": filepath}, ignore_errors=True)

    def measure_file_transfer(self, file_size_mb: int) -> Optional[FileTransferDuration]:
        """Measure time taken to upload and download a file of the specified size"""
        filepath = Path(self.email) / "benchmark" / random_filename(file_size_mb)
        file_bytes = generate_byte_string(file_size_mb)

        try:
            # Delete the file if it already exists
            self.delete_file(str(filepath))

            # Measure the time taken to upload the file
            upload_time = self.upload_file(str(filepath), file_bytes)

            # Measure the time taken to download the file
            download_time = self.download_file(str(filepath))

            return FileTransferDuration(
                upload=upload_time,
                download=download_time,
            )
        except Exception as e:
            print(f"Error during file transfer: {str(e)}")
            return None
        finally:
            # Delete the file after the test
            try:
                self.delete_file(str(filepath))
            except Exception as e:
                print(f"Error deleting file: {str(e)}")

    def get_stats(self, file_size_mb: int, num_runs: int) -> DataTransferStats:
        """Get data transfer statistics for a specific file size"""
        # Collect measurements for each run
        measurements: list[FileTransferDuration] = []
        successful_runs = 0

        for run in range(num_runs):
            try:
                file_transfer_duration = self.measure_file_transfer(file_size_mb)

                if file_transfer_duration is not None:
                    measurements.append(file_transfer_duration)
                    successful_runs += 1

                if run < num_runs - 1:
                    # Wait for a short interval between runs
                    time.sleep(5)
            except Exception as e:
                print(f"Error during run {run + 1}: {str(e)}")
                continue

        if not measurements:
            raise RuntimeError(f"All {num_runs} runs failed. No statistics available.")

        # Calculate statistics from the measurements
        def get_values(attr: str) -> list[float]:
            return [getattr(duration, attr) for duration in measurements]

        return DataTransferStats(
            file_size_mb=file_size_mb,
            upload=Stats.from_values(get_values("upload")),
            download=Stats.from_values(get_values("download")),
            successful_runs=successful_runs,
            total_runs=num_runs,
        )
