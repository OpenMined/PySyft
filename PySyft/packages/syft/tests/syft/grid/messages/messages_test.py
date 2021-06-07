# third party
import pytest

# syft absolute
import syft as sy
from syft.core.common.message import AbstractMessage
from syft.core.io.address import Address
from syft.grid.messages import association_messages
from syft.grid.messages import dataset_messages
from syft.grid.messages import group_messages
from syft.grid.messages import infra_messages
from syft.grid.messages import request_messages
from syft.grid.messages import role_messages
from syft.grid.messages import setup_messages
from syft.grid.messages import tensor_messages

messages = {
    # association_messages
    "DeleteAssociationRequest": {
        "module": association_messages,
        "request_content": {"association_request_id": "21656565"},
        "response_content": {"msg": "Association Request Deleted status_codefully!"},
    },
    "GetAssociationRequest": {
        "module": association_messages,
        "request_content": {"association_request_id": "87564178"},
        "response_content": {
            "entity": "OpenMined",
            "entity-type": "Network",
            "status": "pending",
            "date": "05/12/2022",
        },
    },
    "GetAssociationRequests": {
        "module": association_messages,
        "request_content": {},
        "response_content": {
            "association-requests": [
                {
                    "entity": "OpenMined",
                    "entity-type": "Network",
                    "status": "pending",
                    "date": "05/12/2022",
                },
                {
                    "entity": "Hospital-A",
                    "entity-type": "Domain",
                    "status": "pending",
                    "date": "09/10/2022",
                },
                {
                    "entity": "OpenMined",
                    "entity-type": "Network",
                    "status": "pending",
                    "date": "07/11/2022",
                },
            ]
        },
    },
    "ReceiveAssociationRequest": {
        "module": association_messages,
        "request_content": {},
        "response_content": {},
    },
    "RespondAssociationRequest": {
        "module": association_messages,
        "request_content": {"association_request_id": "87564178", "status": "accept"},
        "response_content": {"msg": "Response registered status_codefully!"},
    },
    "SendAssociationRequest": {
        "module": association_messages,
        "request_content": {
            "domain-name": "My-Domain",
            "domain-address": "http://url:5000",
        },
        "response_content": {"msg": "Association Request Accepted status_codefully!"},
    },
    # dataset_messages
    "CreateDataset": {
        "module": dataset_messages,
        "request_content": {
            "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
            "description": "Dataset Description",
            "tags": ["#x", "#data-sample"],
            "pointable": True,
            "read-permission": ["user-id1", "user-id2", "user-id3"],
            "write-permission": ["user-id1", "user-id5", "user-id9"],
        },
        "response_content": {"msg": "Dataset created succesfully!"},
    },
    "DeleteDataset": {
        "module": dataset_messages,
        "request_content": {"dataset_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Dataset deleted successfully!"},
    },
    "GetDataset": {
        "module": dataset_messages,
        "request_content": {"dataset_id": "eqw9e4a5d846"},
        "response_content": {
            "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
            "description": "Dataset Description",
            "tags": ["#x", "#data-sample"],
            "pointable": True,
            "read-permission": ["user-id1", "user-id2", "user-id3"],
            "write-permission": ["user-id1", "user-id5", "user-id9"],
        },
    },
    "GetDatasets": {
        "module": dataset_messages,
        "request_content": {},
        "response_content": {
            "workers": {
                "626sadaf631": {
                    "dataset": [
                        "<tensor_id>",
                        "<tensor_id>",
                        "<tensor_id>",
                        "<tensor_id>",
                    ],
                    "description": "Dataset Description",
                    "tags": ["#x", "#data-sample"],
                    "pointable": True,
                    "read-permission": ["user-id1", "user-id2", "user-id3"],
                    "write-permission": ["user-id1", "user-id5", "user-id9"],
                },
                "a84ew64wq6e": {
                    "dataset": [
                        "<tensor_id>",
                        "<tensor_id>",
                        "<tensor_id>",
                        "<tensor_id>",
                    ],
                    "description": "Dataset Description",
                    "tags": ["#x", "#data-sample"],
                    "pointable": False,
                    "read-permission": ["user-id1", "user-id2", "user-id3"],
                    "write-permission": [],
                },
            }
        },
    },
    "UpdateDataset": {
        "module": dataset_messages,
        "request_content": {
            "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
            "description": "Dataset Description",
            "tags": ["#x", "#data-sample"],
            "pointable": True,
            "read-permission": ["user-id1", "user-id2", "user-id3"],
            "write-permission": ["user-id1", "user-id5", "user-id9"],
        },
        "response_content": {"msg": "Dataset updated successfully!"},
    },
    # group_messages
    "CreateGroup": {
        "module": group_messages,
        "request_content": {
            "group-name": "Heart diseases group",
            "members": ["user-id1", "user-id2", "user-id3"],
            "data": [
                {"id": "264632213", "permissions": "read"},
                {"id": "264613232", "permissions": "write"},
                {"id": "896632213", "permissions": "read"},
            ],
        },
        "response_content": {"msg": "Group Created Successfully!"},
    },
    "DeleteGroup": {
        "module": group_messages,
        "request_content": {"group_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Group deleted Successfully!"},
    },
    "GetGroup": {
        "module": group_messages,
        "request_content": {"group-id": "eqw9e4a5d846"},
        "response_content": {
            "group-id": "eqw9e4a5d846",
            "group-name": "Heart diseases group",
            "members": ["user-id1", "user-id2", "user-id3"],
            "data": [
                {"id": "264632213", "permissions": "read"},
                {"id": "264613232", "permissions": "write"},
                {"id": "896632213", "permissions": "read"},
            ],
        },
    },
    "GetGroups": {
        "module": group_messages,
        "request_content": {},
        "response_content": {
            "groups": {
                "626sadaf631": {
                    "group-name": "Heart diseases group",
                    "members": ["user-id1", "user-id2", "user-id3"],
                    "data": [
                        {"id": "264632213", "permissions": "read"},
                        {"id": "264613232", "permissions": "write"},
                        {"id": "896632213", "permissions": "read"},
                    ],
                },
                "a84ew64wq6e": {
                    "group-name": "Brain diseases group",
                    "members": ["user-id5", "user-id7", "user-id9"],
                    "data": [
                        {"id": "26463afasd", "permissions": "read"},
                        {"id": "264613dafeqwe", "permissions": "write"},
                        {"id": "896632sdfsf", "permissions": "read"},
                    ],
                },
            }
        },
    },
    "UpdateGroup": {
        "module": group_messages,
        "request_content": {
            "group-id": "eqw9e4a5d846",
            "group-name": "Brain diseases group",
            "members": ["user-id1", "user-id2", "user-id3"],
            "data": [
                {"id": "264632213", "permissions": "read"},
                {"id": "264613232", "permissions": "write"},
                {"id": "896632213", "permissions": "read"},
            ],
        },
        "response_content": {"msg": "Group updated successfully!"},
    },
    # infra_messages
    "CreateWorker": {
        "module": infra_messages,
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
        "module": infra_messages,
        "request_content": {"worker_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Worker Environment deleted Successfully!"},
    },
    "GetWorker": {
        "module": infra_messages,
        "request_content": {"worker-id": "eqw9e4a5d846"},
        "response_content": {
            "worker-id": "eqw9e4a5d846",
            "environment-name": "Heart Diseases Environment",
            "owner": "user-id7",
            "deployment-date": "05/12/2021",
        },
    },
    "GetWorkers": {
        "module": infra_messages,
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
        "module": infra_messages,
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
    # role_messages
    "CreateRole": {
        "module": role_messages,
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
        "module": role_messages,
        "request_content": {"role_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Role has been deleted!"},
    },
    "GetRole": {
        "module": role_messages,
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
        "module": role_messages,
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
        "module": role_messages,
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
    # setup_messages
    "CreateInitialSetUp": {
        "module": setup_messages,
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
        "module": setup_messages,
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
    # tensor_messages
    "CreateTensor": {
        "module": tensor_messages,
        "request_content": {
            "tensor": [1, 2, 3, 4, 5, 6],
            "description": "Tensor Description",
            "tags": ["#x", "#data-sample"],
            "pointable": True,
        },
        "response_content": {"msg": "Tensor created succesfully!"},
    },
    "DeleteTensor": {
        "module": tensor_messages,
        "request_content": {"tensor_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Tensor deleted successfully!"},
    },
    "GetTensor": {
        "module": tensor_messages,
        "request_content": {"tensor_id": "eqw9e4a5d846"},
        "response_content": {
            "description": "Tensor description",
            "tags": ["#x", "#data-sample"],
        },
    },
    "GetTensors": {
        "module": tensor_messages,
        "request_content": {},
        "response_content": {
            "workers": {
                "626sadaf631": {
                    "tensor": [1, 2, 3, 4, 5, 6],
                    "description": "Tensor description",
                    "tags": ["#x", "#data-sample"],
                    "pointable": True,
                },
                "a84ew64wq6e": {
                    "tensor": [9, 8, 2, 3, 5, 6],
                    "description": "Tensor sample description",
                    "tags": ["#y", "#label-sample"],
                    "pointable": True,
                },
            }
        },
    },
    "UpdateTensor": {
        "module": tensor_messages,
        "request_content": {
            "tensor_id": "546a4d51",
            "tensor": [1, 2, 3, 4, 5, 6],
            "description": "Tensor description",
            "tags": ["#x", "#data-sample"],
            "pointable": True,
        },
        "response_content": {"msg": "Tensor updated successfully!"},
    },
    # request_messages
    "CreateRequest": {
        "module": request_messages,
        "request_content": {
            "dataset-id": "68a465aer3adf",
            "user-id": "user-id7",
            "request-type": "read",
        },
        "response_content": {"msg": "Request sent succesfully!"},
    },
    "DeleteRequest": {
        "module": request_messages,
        "request_content": {"request_id": "f2a6as5d16fasd"},
        "response_content": {"msg": "Data Request has been deleted!"},
    },
    "GetRequest": {
        "module": request_messages,
        "request_content": {"request_id": "eqw9e4a5d846"},
        "response_content": {
            "request_id": "asfdaead131",
            "dataset-id": "68a465aer3adf",
            "user-id": "user-id7",
            "request-type": "read",
        },
    },
    "GetRequests": {
        "module": request_messages,
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
        "module": request_messages,
        "request_content": {
            "request_id": "546a4d51",
            "dataset-id": "68a465aer3adf",
            "user-id": "user-id7",
            "request-type": "write",
        },
        "response_content": {"msg": "Data request has been updated successfully!"},
    },
}


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
            "{message_name} does not have a response added to the test configuration"
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
