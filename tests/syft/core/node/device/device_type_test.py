# stdlib
from dataclasses import asdict

# syft absolute
from syft.core.node.device.device_type.device_type import DeviceType
from syft.core.node.device.device_type.specs.cpu import CPU
from syft.core.node.device.device_type.specs.cpu import CpuArchitectureTypes
from syft.core.node.device.device_type.specs.gpu import GPU
from syft.core.node.device.device_type.specs.network import Network
from syft.core.node.device.device_type.specs.provider import Provider
from syft.core.node.device.device_type.specs.storage import Drive
from syft.core.node.device.device_type.specs.storage import DriveType
from syft.core.node.device.device_type.specs.storage import Storage


def test_gpu_device_type() -> None:

    mbs = 16 * 1024 * 1024
    gpu = GPU(
        name="V100",
        count=1,
        manufacturer="NVIDIA",
        memory_total=mbs,
        memory_per_gpu=mbs,
    )

    fixture = {
        "name": "V100",
        "count": 1,
        "manufacturer": "NVIDIA",
        "memory_total": 16777216,
        "memory_per_gpu": 16777216,
    }

    assert fixture == asdict(gpu)


def test_cpu_device_type() -> None:

    cpu = CPU(
        num_cores=8,
        num_threads_per_core=2,
        clock_speed=2.3,
        architecture=CpuArchitectureTypes.x86_64,
    )

    fixture = {
        "num_cores": 8,
        "num_threads_per_core": 2,
        "clock_speed": 2.3,
        "architecture": CpuArchitectureTypes.x86_64,
    }

    assert fixture == asdict(cpu)


def test_network_device_type() -> None:
    net = Network(performance="1000MB")
    fixture = {"performance": "1000MB", "ipv6_supported": False}

    assert fixture == asdict(net)


def test_storage_drives() -> None:
    mbs = 1 * 1024 * 1024 * 1024
    drive = Drive(name="Macintosh HD", storage=mbs)
    raid = Storage(drives=[drive, drive])

    drive_fixture = {
        "name": "Macintosh HD",
        "storage": 1073741824,
        "drive_type": DriveType.SSD,
    }
    raid_fixture = {"drives": [drive_fixture, drive_fixture]}

    assert drive_fixture == asdict(drive)
    assert raid_fixture == asdict(raid)
    assert raid.storage_total == raid.num_drives * mbs
    assert raid.num_drives == 2


def test_provider() -> None:
    print("gfds", str(Provider.USER_OWNED))
    assert Provider.USER_OWNED.value == "user-owned"
    assert Provider.AWS.value == "aws"
    assert Provider.AZURE.value == "azure"
    assert Provider.GCP.value == "gcp"
    assert Provider.GENESIS.value == "genesis"


def test_device_type() -> None:
    cpu = CPU(
        num_cores=8,
        num_threads_per_core=2,
        clock_speed=2.3,
        architecture=CpuArchitectureTypes.x86_64,
    )

    mbs = 1 * 1024 * 1024 * 1024
    drive = Drive(name="Macintosh HD", storage=mbs)
    raid = Storage(drives=[drive, drive])

    macbook_pro_16in_2019 = DeviceType(
        name="MacBook Pro (16-inch, 2019)",
        provider=Provider.USER_OWNED,
        price_per_hour=0,
        memory=32768,
        bare_metal=True,
        storage=raid,
        cpu=cpu,
        gpu=None,
        network=None,
        spot_mode_supported=False,
        price_per_hour_spot=0,
    )

    laptop_fixture = {
        "name": "MacBook Pro (16-inch, 2019)",
        "provider": Provider.USER_OWNED,
        "price_per_hour": 0,
        "memory": 32768,
        "bare_metal": True,
        "storage": {
            "drives": [
                {
                    "name": "Macintosh HD",
                    "storage": 1073741824,
                    "drive_type": DriveType.SSD,
                },
                {
                    "name": "Macintosh HD",
                    "storage": 1073741824,
                    "drive_type": DriveType.SSD,
                },
            ]
        },
        "cpu": {
            "num_cores": 8,
            "num_threads_per_core": 2,
            "clock_speed": 2.3,
            "architecture": CpuArchitectureTypes.x86_64,
        },
        "gpu": None,
        "network": None,
        "spot_mode_supported": False,
        "price_per_hour_spot": 0,
    }

    assert laptop_fixture == asdict(macbook_pro_16in_2019)
