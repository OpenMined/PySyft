from .device_type import DeviceType
from .specs.cpu import CPU, CpuArchitectureTypes
from .specs.provider import Provider
from .specs.storage import Drive, DriveType, Storage

mbp_storage = Storage([Drive(name="/", storage=256000, drive_type=DriveType.SSD)])
mbp_cpu = CPU(
    num_cores=6,
    num_threads_per_core=1,
    clock_speed=2.2,
    architecture=CpuArchitectureTypes.x86_64,
)
macbook_pro_15in_2018 = DeviceType(
    name="Macbook Pro (15-inch, 2018)",
    provider=Provider.USER_OWNED,
    price_per_hour=0,
    memory=32768,
    bare_metal=True,
    storage=mbp_storage,
    cpu=mbp_cpu,
    gpu=None,
    network=None,
    spot_mode_supported=False,
    price_per_hour_spot=0,
)
