# Only the registry lives here at import time. Importing .schema from this
# __init__ would create a cycle (schema imports the model classes, which
# import this package for the registry); the schema side-effect import
# happens in syft_job/__init__.py instead.
from .registry import job_registry

__all__ = [
    "job_registry",
]
