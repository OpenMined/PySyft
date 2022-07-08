# stdlib
import json
import os
import subprocess  # nosec
from typing import Dict as TypeDict
from typing import Optional

# third party
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient

# relative
from .file import user_hagrid_profile

AZURE_SERVICE_PRINCIPAL_PATH = f"{user_hagrid_profile}/azure_sp.json"


class AzureException(Exception):
    pass


def check_azure_authed() -> bool:
    try:
        azure_service_principal()
        return True
    except AzureException as e:
        print(e)

    return False


def login_azure() -> bool:

    cmd = "az login"
    try:
        subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL)  # nosec
        return True
    except Exception:  # nosec
        pass
    return False


def azure_service_principal() -> Optional[TypeDict[str, str]]:
    sp_json = {}
    if not os.path.exists(AZURE_SERVICE_PRINCIPAL_PATH):
        raise AzureException("No service principal so we need to create one first")
    with open(AZURE_SERVICE_PRINCIPAL_PATH, "r") as f:
        sp_json = json.loads(f.read())

    required_keys = ["appId", "displayName", "name", "password", "tenant"]
    for key in required_keys:
        if key not in sp_json:
            raise AzureException(f"{key} missing from {AZURE_SERVICE_PRINCIPAL_PATH}")
    return sp_json


def azure_credentials(
    tenant_id: str, client_id: str, client_secret: str
) -> ClientSecretCredential:
    return ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )


def resource_management_client(
    credentials: ClientSecretCredential, subscription_id: str
) -> ResourceManagementClient:
    return ResourceManagementClient(credentials, subscription_id)
