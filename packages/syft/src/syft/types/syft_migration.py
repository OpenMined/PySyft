# stdlib
from typing import Callable
from typing import Optional
from typing import Union

# relative
from .syft_object import SyftMigrationRegistry
from .transforms import generate_transform_wrapper
from .transforms import validate_klass_and_version


def migrate(
    klass_from: Union[type, str],
    klass_to: Union[type, str],
    version_from: Optional[int] = None,
    version_to: Optional[int] = None,
) -> Callable:
    (
        klass_from_str,
        version_from,
        klass_to_str,
        version_to,
    ) = validate_klass_and_version(
        klass_from=klass_from,
        version_from=version_from,
        klass_to=klass_to,
        version_to=version_to,
    )

    if klass_from_str != klass_to_str:
        raise Exception(
            "Migration can only be performed across classes with same canonical name."
            f"Provided args: klass_from: {klass_from_str}, klass_to: {klass_to_str}"
        )

    if version_from is None or version_to is None:
        raise Exception(
            "Version information missing at either of the classes."
            f"{klass_from_str} has version: {version_from}, {klass_to_str} has version: {version_to}"
        )

    def decorator(function: Callable):
        transforms = function()

        wrapper = generate_transform_wrapper(
            klass_from=klass_from, klass_to=klass_to, transforms=transforms
        )

        SyftMigrationRegistry.register_transform(
            klass_type_str=klass_from_str,
            version_from=version_from,
            version_to=version_to,
            method=wrapper,
        )

        return function

    return decorator
