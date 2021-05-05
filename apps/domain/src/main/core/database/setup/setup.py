# grid relative
from .. import BaseModel
from .. import db


class SetupConfig(BaseModel):
    __tablename__ = "setup"

    id = db.Column(db.Integer(), primary_key=True, autoincrement=True)
    domain_name = db.Column(db.String(255), default="")
    private_key = db.Column(db.String(2048), default="")
    aws_credentials = db.Column(db.String(2048), default="")
    gcp_credentials = db.Column(db.String(2048), default="")
    azure_credentials = db.Column(db.String(2048), default="")
    cache_strategy = db.Column(db.String(255), default="")
    replicate_db = db.Column(db.Boolean(), default=False)
    auto_scale = db.Column(db.Boolean(), default=False)
    tensor_expiration_policy = db.Column(db.Integer(), default=0)
    allow_user_signup = db.Column(db.Boolean(), default=False)

    def __str__(self):
        return f"<Domain Name: {self.domain_name}, Private Key: {self.private_key}, AWS Credentials: {self.aws_credentials}, GCP Credentials: {self.gcp_credentials}, Azure Credentials: {self.azure_credentials}, Cache Strategy: {self.cache_strategy}, Replicate Database: {self.replicate_db}, Auto Scale: {self.auto_scale}, Tensor Exp Policy: {self.tensor_expiration_policy}, Allow User Signup: {self.allow_user_signup}>"


def create_setup(**kwargs):
    new_setup = SetupConfig(**kwargs)
    return new_setup
