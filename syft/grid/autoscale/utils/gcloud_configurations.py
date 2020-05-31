"""Stores enums for all valid configurations in gcloud for easy selection and auto-completion"""
import enum


class MachineType(enum.Enum):
    """Valid values for Machine Types"""

    f1_micro = "f1-micro"


class Zone(enum.Enum):
    """Valid values for Zones"""

    us_central1_a = "us-central1-a"


class ImageFamily(enum.Enum):
    """Valid values for ImageFamaily(OS)"""

    ubuntu_2004_lts = "ubuntu-2004-lts"
