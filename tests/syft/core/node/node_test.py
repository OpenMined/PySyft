import syft as sy


def test_send_message_from_vm_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    assert bob_vm.device is None

    bob_vm_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm_client)
    )


def test_send_message_from_device_client_to_device() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_phone_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_phone_client)
    )


def test_register_vm_on_device() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None

    assert bob_vm_client.device is not None


def test_send_message_from_device_client_to_vm() -> None:
    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_phone_client.register(client=bob_vm_client)

    assert bob_vm.device is not None

    assert bob_vm_client.device is not None

    bob_phone_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm_client)
    )


def test_send_message_from_domain_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()

    bob_phone_client.register(client=bob_vm_client)
    bob_domain_client.register(client=bob_phone_client)

    bob_domain_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm)
    )


def test_send_message_from_network_client_to_vm() -> None:

    bob_vm = sy.VirtualMachine(name="Bob")
    bob_vm_client = bob_vm.get_client()

    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    bob_domain = sy.Domain(name="Bob's Domain")
    bob_domain_client = bob_domain.get_client()

    bob_network = sy.Network(name="Bob's Network")
    bob_network_client = bob_network.get_client()

    bob_phone_client.register(client=bob_vm_client)
    bob_domain_client.register(client=bob_phone_client)
    bob_network_client.register(client=bob_domain_client)

    bob_network_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_vm)
    )
