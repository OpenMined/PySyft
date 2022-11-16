# third party
import pytest

# syft absolute
import syft as sy
from syft.core.common.message import AbstractMessage
from syft.core.io.address import Address
from syft.core.node.common.node_service.infra_manager import infra_manager_messages
from syft.core.node.common.node_service.node_setup import node_setup_messages
from syft.core.node.common.node_service.request_receiver import (
    request_receiver_messages,
)
from syft.core.node.common.node_service.role_manager import role_manager_messages

messages = {
    # infra_manager_messages
    "CreateWorker": {
        "module": infra_manager_messages,
        "request_content": {
            "settings": {
                "instance-size": "t4g.medium",
                "vCPU": "2",
                "network-bandwith": "5Gbps",
                "vGPU": True,
            }
        },
        "response_content": {"msg": "Worker Environment Created Successfully!"},
    },
    "DeleteWorker": {
        "module": infra_manager_messages,
        "request_content": {"worker_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Worker Environment deleted Successfully!"},
    },
    "GetWorker": {
        "module": infra_manager_messages,
        "request_content": {"worker-id": "eqw9e4a5d846"},
        "response_content": {
            "worker-id": "eqw9e4a5d846",
            "environment-name": "Heart Diseases Environment",
            "owner": "user-id7",
            "deployment-date": "05/12/2021",
        },
    },
    "GetWorkers": {
        "module": infra_manager_messages,
        "request_content": {},
        "response_content": {
            "workers": {
                "626sadaf631": {
                    "environment-name": "Heart Diseases Environment",
                    "owner": "user-id7",
                    "deployment-date": "05/12/2021",
                },
                "a84ew64wq6e": {
                    "worker-id": "eqw9e4a5d846",
                    "environment-name": "Brain Diseases Environment",
                    "owner": "user-id8",
                    "deployment-date": "15/12/2021",
                },
            }
        },
    },
    "UpdateWorker": {
        "module": infra_manager_messages,
        "request_content": {
            "worker-id": "eqw9e4a5d846",
            "settings": {
                "instance-size": "t4g.large",
                "vCPU": "2",
                "network-bandwith": "5Gbps",
                "vGPU": True,
            },
        },
        "response_content": {"msg": "Worker Environment updated successfully!"},
    },
    # role_manager_messages
    "CreateRole": {
        "module": role_manager_messages,
        "request_content": {
            "name": "Role Sample",
            "can_triage_results": True,
            "can_edit_settings": False,
            "can_create_users": False,
            "can_edit_roles": False,
            "can_manage_infrastructure": True,
        },
        "response_content": {"msg": "Role created succesfully!"},
    },
    "DeleteRole": {
        "module": role_manager_messages,
        "request_content": {"role_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Role has been deleted!"},
    },
    "GetRole": {
        "module": role_manager_messages,
        "request_content": {"request_id": "eqw9e4a5d846"},
        "response_content": {
            "name": "Role Sample",
            "can_triage_results": True,
            "can_edit_settings": False,
            "can_create_users": False,
            "can_edit_roles": False,
            "can_manage_infrastructure": True,
        },
    },
    "GetRoles": {
        "module": role_manager_messages,
        "request_content": {},
        "response_content": {
            "workers": {
                "626sadaf631": {
                    "name": "Role Sample",
                    "can_triage_results": True,
                    "can_edit_settings": False,
                    "can_create_users": False,
                    "can_edit_roles": False,
                    "can_manage_infrastructure": True,
                },
                "a84ew64wq6e": {
                    "name": "Test Sample",
                    "can_triage_results": False,
                    "can_edit_settings": True,
                    "can_create_users": False,
                    "can_edit_roles": False,
                    "can_manage_infrastructure": False,
                },
            }
        },
    },
    "UpdateRole": {
        "module": role_manager_messages,
        "request_content": {
            "role_id": "9a4f9dasd6",
            "name": "Role Sample",
            "can_triage_results": True,
            "can_edit_settings": False,
            "can_create_users": False,
            "can_edit_roles": False,
            "can_manage_infrastructure": True,
        },
        "response_content": {"msg": "Role has been updated successfully!"},
    },
    # node_setup_messages
    "CreateInitialSetUp": {
        "module": node_setup_messages,
        "request_content": {
            "settings": {
                "cloud-admin-token": "d84we35ad3a1d59a84sd9",
                "cloud-credentials": "<cloud-credentials.pem>",
                "infra": {
                    "autoscaling": True,
                    "triggers": {"memory": "50", "vCPU": "80"},
                },
            }
        },
        "response_content": {"msg": "Initial setup registered successfully!"},
    },
    "GetSetUp": {
        "module": node_setup_messages,
        "request_content": {},
        "response_content": {
            "settings": {
                "cloud-admin-token": "d84we35ad3a1d59a84sd9",
                "cloud-credentials": "<cloud-credentials.pem>",
                "infra": {
                    "autoscaling": True,
                    "triggers": {"memory": "50", "vCPU": "80"},
                },
            }
        },
    },
    # request_receiver_messages
    "CreateRequest": {
        "module": request_receiver_messages,
        "request_content": {
            "dataset-id": "68a465aer3adf",
            "user-id": "user-id7",
            "request-type": "read",
        },
        "response_content": {"msg": "Request sent succesfully!"},
    },
    "DeleteRequest": {
        "module": request_receiver_messages,
        "request_content": {"request_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Data Request has been deleted!"},
    },
    "GetRequest": {
        "module": request_receiver_messages,
        "request_content": {"request_id": "eqw9e4a5d846"},
        "response_content": {
            "request_id": "asfdaead131",
            "dataset-id": "68a465aer3adf",
            "user-id": "user-id7",
            "request-type": "read",
        },
    },
    "GetRequests": {
        "module": request_receiver_messages,
        "request_content": {},
        "response_content": {
            "workers": {
                "626sadaf631": {
                    "dataset-id": "68a465aer3adf",
                    "user-id": "user-id7",
                    "request-type": "read",
                },
                "a84ew64wq6e": {
                    "dataset-id": "98w4e54a6d",
                    "user-id": "user-id9",
                    "request-type": "write",
                },
            }
        },
    },
    "UpdateRequest": {
        "module": request_receiver_messages,
        "request_content": {
            "request_id": "546a4d51",
            "dataset-id": "68a465aer3adf",
            "user-id": "user-id7",
            "request-type": "write",
        },
        "response_content": {"msg": "Data request has been updated successfully!"},
    },
}


# MADHAVA: this needs fixing
@pytest.mark.xfail
@pytest.mark.parametrize("message_name", messages.keys())
def test_message(message_name: str, node: sy.VirtualMachine) -> None:
    content = messages[message_name]
    lib = content["module"]
    request_content = content["request_content"]
    response_content = content["response_content"]
    target = Address(name="Alice")

    msg_func = getattr(lib, message_name + "Message")
    msg = msg_func(content=request_content, address=target, reply_to=target.address)
    message_integrity_test(msg, target)

    if response_content is None:
        pytest.skip(
            f"{message_name} does not have a response added to the test configuration"
        )

    res_func = getattr(lib, message_name + "Response")
    msg = res_func(content=response_content, address=target, status_code=200)
    message_integrity_test(msg, target)


def message_integrity_test(msg: AbstractMessage, target: Address) -> None:
    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2
