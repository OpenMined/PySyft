from syft.core.node.common.node_manager.user_manager import UserBase  # noqa: F401
from syft.core.node.common.node_manager.user_manager import UserCreate  # noqa: F401
from syft.core.node.common.node_manager.user_manager import UserUpdate  # noqa: F401
from syft.core.node.common.node_manager.user_manager import UserInDBBase  # noqa: F401
from syft.core.node.common.node_manager.user_manager import User  # noqa: F401
from syft.core.node.common.node_manager.user_manager import UserInDB  # noqa: F401

# NOTE: This functionality was moved to PySyft so that data can Users can be loaded into
# the user database even when a Domain object is instantiated in a Notebook or unit test for the
# purpose of dev or testing (aka the object is used independently of PyGrid). Relevant CRUD functionality
# may make its way over to PySyft in due course for same/similar reasons in the near future. If you'd like
# to chat about this, let's hop on a call! - Andrew

# # stdlib
# from typing import Optional
#
# # third party
# from pydantic import BaseModel
# from pydantic import EmailStr
#
#
# # Shared properties
# class UserBase(BaseModel):
#     email: Optional[EmailStr] = None
#     is_active: Optional[bool] = True
#     is_superuser: bool = False
#     full_name: Optional[str] = None
#
#
# # Properties to receive via API on creation
# class UserCreate(UserBase):
#     email: EmailStr
#     password: str
#
#
# # Properties to receive via API on update
# class UserUpdate(UserBase):
#     password: Optional[str] = None
#
#
# class UserInDBBase(UserBase):
#     id: Optional[int] = None
#
#     class Config:
#         orm_mode = True
#
#
# # Additional properties to return via API
# class User(UserInDBBase):
#     pass
#
#
# # Additional properties stored in DB
# class UserInDB(UserInDBBase):
#     hashed_password: str
