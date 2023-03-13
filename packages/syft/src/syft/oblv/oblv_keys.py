# # relative
# from .syft_object import SYFT_OBJECT_VERSION_1
# from .syft_object import SyftObject


# class NoSQLOblvKeys(SyftObject):
#     # version
#     __canonical_name__ = "OldOblvKeys"
#     __version__ = SYFT_OBJECT_VERSION_1

#     # fields
#     public_key: bytes
#     private_key: bytes

#     # serde / storage rules
#     __attr_state__ = ["public_key", "private_key"]
#     __attr_searchable__ = []
#     __attr_unique__ = ["private_key"]
