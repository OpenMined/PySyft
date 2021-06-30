# third party
from sqlalchemy import Boolean
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

# relative
from . import Base


class SetupConfig(Base):
    __tablename__ = "setup"

    id = Column(Integer(), primary_key=True, autoincrement=True)
    domain_name = Column(String(255), default="")
    private_key = Column(String(2048), default="")
    aws_credentials = Column(String(2048), default="")
    gcp_credentials = Column(String(2048), default="")
    azure_credentials = Column(String(2048), default="")
    cache_strategy = Column(String(255), default="")
    replicate_db = Column(Boolean(), default=False)
    auto_scale = Column(Boolean(), default=False)
    tensor_expiration_policy = Column(Integer(), default=0)
    allow_user_signup = Column(Boolean(), default=False)

    def __str__(self):
        return f"<Domain Name: {self.domain_name}, Private Key: {self.private_key}, AWS Credentials: {self.aws_credentials}, GCP Credentials: {self.gcp_credentials}, Azure Credentials: {self.azure_credentials}, Cache Strategy: {self.cache_strategy}, Replicate Database: {self.replicate_db}, Auto Scale: {self.auto_scale}, Tensor Exp Policy: {self.tensor_expiration_policy}, Allow User Signup: {self.allow_user_signup}>"


def create_setup(**kwargs):
    new_setup = SetupConfig(**kwargs)
    return new_setup
