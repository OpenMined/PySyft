import hashlib
import json
import os

from magika import Magika


# Class to hold file metadata
class FileInfo:
    def __init__(self, hash_str, extension, size_mb, mime_type, magika_group):
        self.hash_str: str = hash_str
        self.extension: str = extension
        self.size_mb: float = size_mb
        self.mime_type: str = mime_type
        self.magika_group: str = magika_group

    def to_dict(self):
        return {
            "hash_str": self.hash_str,
            "extension": self.extension,
            "size_mb": self.size_mb,
            "mime_type": self.mime_type,
            "magika_group": self.magika_group,
        }


ignore_files = [".DS_Store", "_.syftperm"]
ignore_extensions = [".pyc"]


def read_file_as_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()


# Function to calculate SHA-256 hash of a file
def calculate_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ends_with(filename):
    for ext in ignore_extensions:
        if filename.endswith(ext):
            return True
    return False


# Function to list all files recursively and gather metadata
def list_files_with_metadata(directory_path, cache_file="./file_metadata_cache.json"):
    # Load cached metadata if it exists
    cached_data = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as cache:
            cached_data = json.load(cache)

    magika = Magika()
    result = {}

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file in ignore_files or ends_with(file):
                continue
            file_path = os.path.join(root, file)
            file_hash = calculate_hash(file_path)

            # Check if the file is in the cache and unchanged
            if (
                file_path in cached_data
                and cached_data[file_path]["hash_str"] == file_hash
            ):
                result[file_path] = cached_data[file_path]
            else:
                # Calculate new metadata
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                extension = os.path.splitext(file)[-1].lstrip(".")
                file_bytes = read_file_as_bytes(file_path)
                mime_info = magika.identify_bytes(file_bytes)

                file_info = FileInfo(
                    hash_str=file_hash,
                    extension=extension,
                    size_mb=size_mb,
                    mime_type=mime_info.output.mime_type,
                    magika_group=mime_info.output.group,
                )
                result[file_path] = file_info.to_dict()

    # Save updated metadata to cache
    with open(cache_file, "w") as cache:
        json.dump(result, cache, indent=4)

    return result
