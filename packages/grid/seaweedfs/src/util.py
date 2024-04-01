# stdlib
from hashlib import sha256


def seaweed_config_name(prefix: str, bucket_name: str) -> str:
    """Seaweed-friendly name for the remote config"""
    return prefix + sha256(bucket_name.encode()).hexdigest()[:8]
