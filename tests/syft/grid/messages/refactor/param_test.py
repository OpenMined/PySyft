import pytest
import importlib

import syft as sy
from syft import serialize
from syft.core.io.address import Address

from syft.grid.messages import association_messages
from syft.grid.messages import dataset_messages
from syft.grid.messages import group_messages
from syft.grid.messages import infra_messages
from syft.grid.messages import role_messages
from syft.grid.messages import setup_messages
from syft.grid.messages import tensor_messages
from syft.grid.messages import request_messages 

messages = [
    # association_messages
    ("DeleteAssociationRequest", association_messages),
    ("GetAssociationRequest", association_messages),
    ("GetAssociationRequests", association_messages),
    ("ReceiveAssociationRequest", association_messages),
    ("RespondAssociationRequest", association_messages),
    ("SendAssociationRequest", association_messages),

    # dataset_messages
    ("CreateDataset", dataset_messages),
    ("DeleteDataset", dataset_messages),
    ("GetDataset", dataset_messages),
    ("GetDatasets", dataset_messages),
    ("UpdateDataset", dataset_messages),

    # group_messages
    ("CreateGroup", group_messages),
    ("DeleteGroup", group_messages),
    ("GetGroup", group_messages),
    ("GetGroups", group_messages),
    ("UpdateGroup", group_messages),

    # infra_messages
    ("CreateWorker", infra_messages),
    ("DeleteWorker", infra_messages),
    ("GetWorker", infra_messages),
    ("GetWorkers", infra_messages),
    ("UpdateWorker", infra_messages),

    # role_messages
    ("CreateRole", role_messages),
    ("DeleteRole", role_messages),
    ("GetRole", role_messages),
    ("GetRoles", role_messages),
    ("UpdateRole", role_messages),

    # setup_messages
    ("CreateInitialSetUp", setup_messages),
    ("GetSetUp", setup_messages),

    # tensor_messages
    ("CreateTensor", tensor_messages),
    ("DeleteTensor", tensor_messages),
    ("GetTensor", tensor_messages),
    ("GetTensors", tensor_messages),
    ("UpdateTensor", tensor_messages),

    # request_messages
    ("CreateRequest", request_messages),
    ("DeleteRequest", request_messages),
    ("GetRequest", request_messages),
    ("GetRequests", request_messages),
    ("UpdateRequest", request_messages),
]


request_contents = [
    # association_messages
    {"association_request_id": "21656565"},
    {"association_request_id": "87564178"},
    {},
    {},
    {"association_request_id": "87564178", "status": "accept"},
    {"domain-name": "My-Domain", "domain-address": "http://url:5000"},
    
    # dataset_messages
    {
        "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
        "description": "Dataset Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
        "read-permission": ["user-id1", "user-id2", "user-id3"],
        "write-permission": ["user-id1", "user-id5", "user-id9"],
    },
    {"dataset_id": "f2a6as5d16fasd"},
    {"dataset_id": "eqw9e4a5d846"},
    {},
    {
        "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
        "description": "Dataset Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
        "read-permission": ["user-id1", "user-id2", "user-id3"],
        "write-permission": ["user-id1", "user-id5", "user-id9"],
    },

    # group_messages
    {
        "group-name": "Heart diseases group",
        "members": ["user-id1", "user-id2", "user-id3"],
        "data": [
            {"id": "264632213", "permissions": "read"},
            {"id": "264613232", "permissions": "write"},
            {"id": "896632213", "permissions": "read"},
        ],
    },
    {"group_id": "f2a6as5d16fasd"},
    {"group-id": "eqw9e4a5d846"},
    {},
    {
        "group-id": "eqw9e4a5d846",
        "group-name": "Brain diseases group",
        "members": ["user-id1", "user-id2", "user-id3"],
        "data": [
            {"id": "264632213", "permissions": "read"},
            {"id": "264613232", "permissions": "write"},
            {"id": "896632213", "permissions": "read"},
        ],
    },

    # infra_messages
    {
        "settings": {
            "instance-size": "t4g.medium",
            "vCPU": "2",
            "network-bandwith": "5Gbps",
            "vGPU": True,
        }
    },
    {"worker_id": "f2a6as5d16fasd"},
    {"worker-id": "eqw9e4a5d846"},
    {},
    {
        "worker-id": "eqw9e4a5d846",
        "settings": {
            "instance-size": "t4g.large",
            "vCPU": "2",
            "network-bandwith": "5Gbps",
            "vGPU": True,
        },
    },

    # role_messages
    {
        "name": "Role Sample",
        "can_triage_results": True,
        "can_edit_settings": False,
        "can_create_users": False,
        "can_edit_roles": False,
        "can_manage_infrastructure": True,
    },
    {"role_id": "f2a6as5d16fasd"},
    {"request_id": "eqw9e4a5d846"},
    {},
    {
        "role_id": "9a4f9dasd6",
        "name": "Role Sample",
        "can_triage_results": True,
        "can_edit_settings": False,
        "can_create_users": False,
        "can_edit_roles": False,
        "can_manage_infrastructure": True,
    },

    # setup_messages
    {
        "settings": {
            "cloud-admin-token": "d84we35ad3a1d59a84sd9",
            "cloud-credentials": "<cloud-credentials.pem>",
            "infra": {"autoscaling": True, "triggers": {"memory": "50", "vCPU": "80"}},
        }
    },
    {},

    # tensor_messages
    {
        "tensor": [1, 2, 3, 4, 5, 6],
        "description": "Tensor Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
    },
    {"tensor_id": "f2a6as5d16fasd"},
    {"tensor_id": "eqw9e4a5d846"},
    {},
    {
        "tensor_id": "546a4d51",
        "tensor": [1, 2, 3, 4, 5, 6],
        "description": "Tensor description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
    },

    # request_messages
    {
        "dataset-id": "68a465aer3adf",
        "user-id": "user-id7",
        "request-type": "read",
    },
    {"request_id": "f2a6as5d16fasd"},
    {"request_id": "eqw9e4a5d846"},
    {},
    {
        "request_id": "546a4d51",
        "dataset-id": "68a465aer3adf",
        "user-id": "user-id7",
        "request-type": "write",
    }
]


response_contents = [
    # association_messages
    {"msg": "Association Request Deleted status_codefully!"},
    {
        "entity": "OpenMined",
        "entity-type": "Network",
        "status": "pending",
        "date": "05/12/2022",
    },
    {
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
    {},
    {"msg": "Response registered status_codefully!"},
    {"msg": "Association Request Accepted status_codefully!"},
    
    # dataset_messages
    {"msg": "Dataset created succesfully!"},
    {"msg": "Dataset deleted successfully!"},
    {
        "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
        "description": "Dataset Description",
        "tags": ["#x", "#data-sample"],
        "pointable": True,
        "read-permission": ["user-id1", "user-id2", "user-id3"],
        "write-permission": ["user-id1", "user-id5", "user-id9"],
    },
    {
        "workers": {
            "626sadaf631": {
                "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
                "description": "Dataset Description",
                "tags": ["#x", "#data-sample"],
                "pointable": True,
                "read-permission": ["user-id1", "user-id2", "user-id3"],
                "write-permission": ["user-id1", "user-id5", "user-id9"],
            },
            "a84ew64wq6e": {
                "dataset": ["<tensor_id>", "<tensor_id>", "<tensor_id>", "<tensor_id>"],
                "description": "Dataset Description",
                "tags": ["#x", "#data-sample"],
                "pointable": False,
                "read-permission": ["user-id1", "user-id2", "user-id3"],
                "write-permission": [],
            },
        }
    },
    {"msg": "Dataset updated successfully!"},

    # group_messages
    {"msg": "Group Created Successfully!"},
    {"msg": "Group deleted Successfully!"},
    {
        "group-id": "eqw9e4a5d846",
        "group-name": "Heart diseases group",
        "members": ["user-id1", "user-id2", "user-id3"],
        "data": [
            {"id": "264632213", "permissions": "read"},
            {"id": "264613232", "permissions": "write"},
            {"id": "896632213", "permissions": "read"},
        ],
    },
    {
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
    {"msg": "Group updated successfully!"},

    # infra_messages
    {"msg": "Worker Environment Created Successfully!"},
    {"msg": "Worker Environment deleted Successfully!"},
    {
        "worker-id": "eqw9e4a5d846",
        "environment-name": "Heart Diseases Environment",
        "owner": "user-id7",
        "deployment-date": "05/12/2021",
    },
    {
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
    {"msg": "Worker Environment updated successfully!"},

    # roles_messages

    {"msg": "Role created succesfully!"},
    {"msg": "Role has been deleted!"},
    {
        "name": "Role Sample",
        "can_triage_results": True,
        "can_edit_settings": False,
        "can_create_users": False,
        "can_edit_roles": False,
        "can_manage_infrastructure": True,
    },
    {
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
    {"msg": "Role has been updated successfully!"},

    # setup_messages
    {"msg": "Initial setup registered successfully!"},
    {
        "settings": {
            "cloud-admin-token": "d84we35ad3a1d59a84sd9",
            "cloud-credentials": "<cloud-credentials.pem>",
            "infra": {"autoscaling": True, "triggers": {"memory": "50", "vCPU": "80"}},
        }
    },

    # tensor_messages
    {"msg": "Tensor created succesfully!"},
    {"msg": "Tensor deleted successfully!"},
    {
        "description": "Tensor description",
        "tags": ["#x", "#data-sample"],
    },
    {
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
    {"msg": "Tensor updated successfully!"},

    # request_messages
    {"msg": "Request sent succesfully!"},
    {"msg": "Data Request has been deleted!"},
    {
        "request_id": "asfdaead131",
        "dataset-id": "68a465aer3adf",
        "user-id": "user-id7",
        "request-type": "read",
    },
    {
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
    {"msg": "Data request has been updated successfully!"}
]

@pytest.mark.parametrize("iterator", range(len(messages)))
def test(iterator, node: sy.VirtualMachine) -> None:
    request, lib = messages[iterator]
    request_content = request_contents[iterator]
    response_content = response_contents[iterator]
    target = Address(name="Alice")

    msg_func = getattr(lib, request+"Message")
    msg = msg_func(content=request_content, address=target, reply_to=target.address)
    actual_test(msg, target)

    if response_content == None:
        pytest.skip("{request} does not have a response added to test configuration")

    res_func = getattr(lib, request+"Response")
    msg = res_func(content=response_content, address=target, status_code=200)
    actual_test(msg, target)


def actual_test(msg, target):
    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == target
    assert msg.content == msg2.content
    assert msg == msg2