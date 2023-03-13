# # This test code was written by the `hypothesis.extra.ghostwriter` module
# # and is provided under the Creative Commons Zero public domain dedication.

# # stdlib
# import typing

# # third party
# from hypothesis import given
# from hypothesis import strategies as st
# from pydantic._hypothesis_plugin import is_valid_email
# import pydantic.networks

# # syft absolute
# from syft import UID
# import syft.core.common.uid
# import syft.core.node.common.node_table.syft_object
# import syft.core.node.new.credentials
# import syft.core.node.new.node
# from syft.core.node.new.node import NewNode
# import syft.core.node.new.transforms
# import syft.core.node.new.user
# from syft.core.node.new.user import SyftSigningKey
# from syft.core.node.new.user import TransformContext


# @given(
#     id=st.one_of(st.none(), st.builds(UID)),
#     email=st.one_of(st.emails().filter(is_valid_email)),
#     name=st.one_of(st.none(), st.text()),
#     hashed_password=st.one_of(st.none(), st.text()),
#     salt=st.one_of(st.none(), st.text()),
#     signing_key=st.one_of(st.none(), st.builds(SyftSigningKey.generate)),
#     verify_key=st.one_of(st.none(), st.just(SyftSigningKey.generate().verify_key)),
#     role=st.one_of(st.none(), st.sampled_from(syft.core.node.new.user.ServiceRole)),
#     institution=st.one_of(st.none(), st.text()),
#     website=st.one_of(st.none(), st.text()),
#     created_at=st.one_of(st.none(), st.text()),
# )
# def test_fuzz_User(
#     id: typing.Optional[syft.core.common.uid.UID],
#     email: typing.Optional[pydantic.networks.EmailStr],
#     name: typing.Optional[str],
#     hashed_password: typing.Optional[str],
#     salt: typing.Optional[str],
#     signing_key: typing.Optional[syft.core.node.new.credentials.SyftSigningKey],
#     verify_key: typing.Optional[syft.core.node.new.credentials.SyftVerifyKey],
#     role: typing.Optional[syft.core.node.new.user.ServiceRole],
#     institution: typing.Optional[str],
#     website: typing.Optional[str],
#     created_at: typing.Optional[str],
# ) -> None:
#     syft.core.node.new.user.User(
#         id=id,
#         email=email,
#         name=name,
#         hashed_password=hashed_password,
#         salt=salt,
#         signing_key=signing_key,
#         verify_key=verify_key,
#         role=role,
#         institution=institution,
#         website=website,
#         created_at=created_at,
#     )


# @given(v=st.emails().filter(is_valid_email))
# def test_fuzz_User_make_email(v: pydantic.networks.EmailStr) -> None:
#     syft.core.node.new.user.User.make_email(v=v)


# @given(
#     id=st.one_of(st.none(), st.builds(UID)),
#     email=st.emails().filter(is_valid_email),
#     name=st.text(),
#     role=st.one_of(st.none(), st.sampled_from(syft.core.node.new.user.ServiceRole)),
#     password=st.text(),
#     password_verify=st.text(),
#     verify_key=st.one_of(st.none(), st.just(SyftSigningKey.generate().verify_key)),
#     institution=st.one_of(st.none(), st.text()),
#     website=st.one_of(st.none(), st.text()),
# )
# def test_fuzz_UserCreate(
#     id: typing.Optional[syft.core.common.uid.UID],
#     email: pydantic.networks.EmailStr,
#     name: str,
#     role: typing.Optional[syft.core.node.new.user.ServiceRole],
#     password: str,
#     password_verify: str,
#     verify_key: typing.Optional[syft.core.node.new.credentials.SyftVerifyKey],
#     institution: typing.Optional[str],
#     website: typing.Optional[str],
# ) -> None:
#     syft.core.node.new.user.UserCreate(
#         id=id,
#         email=email,
#         name=name,
#         role=role,
#         password=password,
#         password_verify=password_verify,
#         verify_key=verify_key,
#         institution=institution,
#         website=website,
#     )


# @given(
#     id=st.builds(UID),
#     email=st.text(),
#     signing_key=st.builds(SyftSigningKey.generate),
# )
# def test_fuzz_UserPrivateKey(
#     id: syft.core.common.uid.UID,
#     email: str,
#     signing_key: syft.core.node.new.credentials.SyftSigningKey,
# ) -> None:
#     syft.core.node.new.user.UserPrivateKey(id=id, email=email, signing_key=signing_key)


# @given(
#     id=st.one_of(
#         st.none(),
#         st.builds(UID),
#     ),
#     email=st.one_of(st.none(), st.emails().filter(is_valid_email)),
#     verify_key=st.one_of(st.none(), st.just(SyftSigningKey.generate().verify_key)),
#     name=st.one_of(st.none(), st.text()),
# )
# def test_fuzz_UserSearch(
#     id: typing.Optional[syft.core.common.uid.UID],
#     email: typing.Optional[pydantic.networks.EmailStr],
#     verify_key: typing.Optional[syft.core.node.new.credentials.SyftVerifyKey],
#     name: typing.Optional[str],
# ) -> None:
#     syft.core.node.new.user.UserSearch(
#         id=id, email=email, verify_key=verify_key, name=name
#     )


# @given(
#     id=st.one_of(
#         st.none(),
#         st.builds(UID),
#     ),
#     email=st.one_of(st.none(), st.emails().filter(is_valid_email)),
#     name=st.one_of(st.none(), st.text()),
#     role=st.one_of(st.none(), st.sampled_from(syft.core.node.new.user.ServiceRole)),
#     password=st.one_of(st.none(), st.text()),
#     password_verify=st.one_of(st.none(), st.text()),
#     verify_key=st.one_of(st.none(), st.just(SyftSigningKey.generate().verify_key)),
#     institution=st.one_of(st.none(), st.text()),
#     website=st.one_of(st.none(), st.text()),
# )
# def test_fuzz_UserUpdate(
#     id: typing.Optional[syft.core.common.uid.UID],
#     email: typing.Optional[pydantic.networks.EmailStr],
#     name: typing.Optional[str],
#     role: typing.Optional[syft.core.node.new.user.ServiceRole],
#     password: typing.Optional[str],
#     password_verify: typing.Optional[str],
#     verify_key: typing.Optional[syft.core.node.new.credentials.SyftVerifyKey],
#     institution: typing.Optional[str],
#     website: typing.Optional[str],
# ) -> None:
#     syft.core.node.new.user.UserUpdate(
#         id=id,
#         email=email,
#         name=name,
#         role=role,
#         password=password,
#         password_verify=password_verify,
#         verify_key=verify_key,
#         institution=institution,
#         website=website,
#     )


# @given(v=st.emails().filter(is_valid_email))
# def test_fuzz_UserUpdate_make_email(v: pydantic.networks.EmailStr) -> None:
#     syft.core.node.new.user.UserUpdate.make_email(v=v)


# @given(
#     id=st.one_of(
#         st.none(),
#         st.builds(UID),
#     ),
#     email=st.one_of(st.none(), st.emails().filter(is_valid_email)),
#     name=st.one_of(st.none(), st.text()),
#     role=st.one_of(st.none(), st.sampled_from(syft.core.node.new.user.ServiceRole)),
#     password=st.one_of(st.none(), st.text()),
#     password_verify=st.one_of(st.none(), st.text()),
#     verify_key=st.one_of(st.none(), st.just(SyftSigningKey.generate().verify_key)),
#     institution=st.one_of(st.none(), st.text()),
#     website=st.one_of(st.none(), st.text()),
# )
# def test_fuzz_UserView(
#     id: typing.Optional[syft.core.common.uid.UID],
#     email: typing.Optional[pydantic.networks.EmailStr],
#     name: typing.Optional[str],
#     role: typing.Optional[syft.core.node.new.user.ServiceRole],
#     password: typing.Optional[str],
#     password_verify: typing.Optional[str],
#     verify_key: typing.Optional[syft.core.node.new.credentials.SyftVerifyKey],
#     institution: typing.Optional[str],
#     website: typing.Optional[str],
# ) -> None:
#     syft.core.node.new.user.UserView(
#         id=id,
#         email=email,
#         name=name,
#         role=role,
#         password=password,
#         password_verify=password_verify,
#         verify_key=verify_key,
#         institution=institution,
#         website=website,
#     )


# # @given(password=st.text(), hashed_password=st.text())
# # def test_fuzz_check_pwd(password: str, hashed_password: str) -> None:
# #     syft.core.node.new.user.check_pwd(
# #         password=password, hashed_password=hashed_password
# #     )


# @given(role=st.sampled_from(syft.core.node.new.user.ServiceRole))
# def test_fuzz_default_role(role: syft.core.node.new.user.ServiceRole) -> None:
#     syft.core.node.new.user.default_role(role=role)


# @given(list_keys=st.lists(st.text()))
# def test_fuzz_drop(list_keys: typing.List[str]) -> None:
#     syft.core.node.new.user.drop(list_keys=list_keys)


# @given(context=st.builds(TransformContext, output=st.from_type(dict)))
# def test_fuzz_generate_key(
#     context: syft.core.node.new.transforms.TransformContext,
# ) -> None:
#     syft.core.node.new.user.generate_key(context=context)


# @given(
#     context=st.builds(
#         TransformContext,
#         output=st.just({"password": "iambatman", "password_verify": "iambatman"}),
#         node=st.one_of(st.none(), st.from_type(NewNode)),
#         credentials=st.one_of(st.none(), st.just(SyftSigningKey.generate().verify_key)),
#     )
# )
# def test_fuzz_hash_password(
#     context: syft.core.node.new.transforms.TransformContext,
# ) -> None:
#     syft.core.node.new.user.hash_password(context=context)


# @given(list_keys=st.lists(st.text()))
# def test_fuzz_keep(list_keys: typing.List[str]) -> None:
#     syft.core.node.new.user.keep(list_keys=list_keys)


# @given(
#     context=st.builds(
#         TransformContext,
#         output=st.just({"email": "iambatman@wayne.inc"}),
#         node=st.one_of(st.none(), st.from_type(NewNode)),
#         credentials=st.one_of(st.none(), st.just(SyftSigningKey.generate().verify_key)),
#     )
# )
# def test_fuzz_validate_email(
#     context: syft.core.node.new.transforms.TransformContext,
# ) -> None:
#     syft.core.node.new.user.validate_email(context=context)
