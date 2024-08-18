# stdlib
import ast
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from random import choice

# third party
from faker import Faker
import pandas as pd
from tqdm import tqdm

# syft absolute
import syft as sy
from syft import autocache
from syft.client.client import SyftClient
from syft.service.api.api import register_fn_in_linecache
from syft.service.code.user_code import syft_function_single_use
from syft.service.project.project import Project
from syft.service.response import SyftError
from syft.service.user.user import UserView
from syft.service.user.user_roles import ServiceRole
from syft.types.base import SyftBaseModel


class BaseFixtureConfig(SyftBaseModel):
    class Config:
        arbitrary_types_allowed = True

    to_create: int
    created: int = 0


class DatasetType(Enum):
    """Type of datasets one can create."""

    IMAGE = "image"
    TABULAR = "tabular"


class UserFixtureConfig(BaseFixtureConfig):
    role: ServiceRole


class DatasetFixtureConfig(BaseFixtureConfig):
    type: DatasetType


class CodeFixtureConfig(BaseFixtureConfig):
    via_project: bool
    # uni_distribution: bool


@dataclass
class UserEmailPassword:
    email: str
    password: str


class FixtureConfig(SyftBaseModel):
    user: list[UserFixtureConfig]
    dataset: list[DatasetFixtureConfig]
    user_code: CodeFixtureConfig


class SyftFixture:
    """
    A class to create sample data for the syft platform.
    """

    def __init__(self, config: dict, root_client: SyftClient) -> None:
        self.config = FixtureConfig(**config)
        self.root_client = root_client
        self.faker = Faker()

    def _add_users(self):
        for user_config in tqdm(self.config.user, desc="Users", position=0):
            users_to_create = user_config.to_create - user_config.created
            for _ in range(users_to_create):
                self._add_user(role=user_config.role)
                user_config.created += 1

    def _add_user(self, role: ServiceRole):
        password = self.faker.password()

        user = self.root_client.api.services.user.create(
            name=self.faker.name(),
            email=self.faker.email(),
            password=password,
            password_verify=password,
            institution=self.faker.company(),
            website=self.faker.url(),
            role=role,
        )
        assert isinstance(user, UserView)

    @staticmethod
    def _get_sample_tabular_data(split_ratio: float = 0.8):
        tabular_data_url = autocache(
            "https://github.com/OpenMined/datasets/blob/main/trade_flow/ca%20-%20feb%202021.csv?raw=True"
        )
        tabular_data = pd.read_csv(tabular_data_url)
        stringcols = tabular_data.select_dtypes(include="object").columns
        tabular_data[stringcols] = tabular_data[stringcols].fillna("").astype(str)
        columns = tabular_data.shape[0]
        private_index = int(split_ratio * columns)
        private_data = tabular_data[:private_index]
        mock_data = tabular_data[private_index:]
        return private_data, mock_data

    def _get_sample_data(self, type: DatasetType) -> tuple:
        if type == DatasetType.TABULAR:
            return self._get_sample_tabular_data()
        else:
            raise NotImplementedError

    def _add_dataset(self, data_type: DatasetType):
        dataset_name = f"{self.faker.first_name()}-Dataset"
        private_data, mock_data = self._get_sample_data(data_type)
        asset = sy.Asset(
            name=f"{dataset_name}-{self.faker.uuid4()[:6]}",
            description=self.faker.text(),
            data=private_data,
            mock=mock_data,
        )

        dataset = sy.Dataset(
            name=dataset_name,
            description=self.faker.text(),
            url=self.faker.url(),
            asset_list=[asset],
        )
        res = self.root_client.upload_dataset(dataset)
        assert not isinstance(res, sy.SyftError)

    def _add_datasets(self):
        for dataset_config in tqdm(self.config.dataset, desc="Datasets:", position=0):
            datasets_to_create = dataset_config.to_create - dataset_config.created
            for _ in range(datasets_to_create):
                self._add_dataset(data_type=dataset_config.type)
                dataset_config.created += 1

    def _gen_sample_func(self, syft_decorator: Callable):
        func_name = self.faker.pystr(min_chars=None, max_chars=12)
        func_str = f'def {func_name}() -> str:\n    return "Hello -> {func_name}"\n'
        src = ast.unparse(ast.parse(func_str).body[0])
        raw_byte_code = compile(src, func_name, "exec")
        register_fn_in_linecache(func_name, src)
        exec(raw_byte_code)
        new_func = eval(func_name, None, locals())
        return syft_decorator()(new_func)

    def _submit_user_code(self, via_project: bool, ds_client: SyftClient):
        new_func = self._gen_sample_func(syft_decorator=syft_function_single_use)
        if via_project:
            new_project = sy.Project(
                name=f"Project-{self.faker.name()}",
                description=self.faker.text(),
                members=[ds_client],
            )
            res = new_project.create_code_request(new_func, ds_client)
            project = new_project.send()
            assert isinstance(project, Project)
        else:
            res = self.root_client.code.request_code_execution(
                new_func, reason=self.faker.text()
            )
            assert not isinstance(res, SyftError), res

    def _add_user_code(self):
        ds_users = self.root_client.users.search(role=ServiceRole.DATA_SCIENTIST)

        user_client_map = {}

        if len(ds_users) == 0:
            print(
                "No Data scientist available to add user code to. "
                "Please create some users with Data Scientist role."
            )
            return

        user_code_to_create = (
            self.config.user_code.to_create - self.config.user_code.created
        )
        for _ in tqdm(range(user_code_to_create), desc="User Code", position=1):
            # Randomly choose a data scientist
            ds_user = choice(ds_users)
            if ds_user.email not in user_client_map:
                user_client_map[ds_user.email] = self.root_client.login_as(
                    email=ds_user.email
                )

            # Get the DS client
            ds_client = user_client_map[ds_user.email]

            # Create user code
            self._submit_user_code(
                self.config.user_code.via_project,
                ds_client=ds_client,
            )
            self.config.user_code.created += 1

    def create(self) -> str:
        self._add_users()
        self._add_datasets()
        self._add_user_code()
        print(self.info())

    def info(self):
        _repr_ = "\nUsers: "

        for user_conf in self.config.user:
            _repr_ += (
                f"\n\t{user_conf.role.name}: {user_conf.created}/{user_conf.to_create}"
            )

        _repr_ += "\nDatasets:"
        for dataset_conf in self.config.dataset:
            _repr_ += f"\n\t{dataset_conf.type.name}: {dataset_conf.created}/{dataset_conf.to_create}"

        user_code_conf = self.config.user_code
        _repr_ += f"\nUserCode: {user_code_conf.created}/{user_code_conf.to_create}"
        _repr_ += f"\n\tVia Project: {user_code_conf.via_project}, Distribution: Random"

        print(_repr_)
