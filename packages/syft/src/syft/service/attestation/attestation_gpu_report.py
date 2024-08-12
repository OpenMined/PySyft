# stdlib
import time
from typing import Any

GPU_ATTESTATION_SUMMARY_TEMPLATE = """
-----------------------------------------------------------
📝 Attestation Report Summary
-----------------------------------------------------------
Issued At: {Issued At}
Valid From: {Valid From}
Expiry: {Expiry} (Token expires in: {Remaining Time})

📢 Issuer Information
-----------------------------------------------------------
Issuer: {Issuer}
Attestation Type: {Attestation Type}
Device ID: {Device ID}

🔒 Security Features
-----------------------------------------------------------
Secure Boot: {Secure Boot}
Debugging: {Debugging}

💻 Hardware
-----------------------------------------------------------
HW Model : {HW Model}
OEM ID: {OEM ID}
Driver Version: {Driver Version}
VBIOS Version: {VBIOS Version}
"""


class GPUAttestationReport:
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.expected_values = {
            "secboot": True,
            "dbgstat": "disabled",
        }

    def check(self, field_name: str) -> Any:
        actual_value = self.get_nested_value(self.report, field_name)
        expected_value = self.expected_values.get(field_name, None)
        return actual_value == expected_value

    def status(self, field_name: str) -> str:
        return "✅" if self.check(field_name) else "❌"

    def is_secure(self) -> bool:
        return all(self.check(field_name) for field_name in self.expected_values.keys())

    def get_nested_value(self, data: dict, key: str) -> Any:
        keys = key.split(".")
        for k in keys:
            data = data.get(k, {})
        return data

    def generate_summary(self) -> str:
        attestation_summary = {
            "Issued At": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(self.report["iat"])
            ),
            "Valid From": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(self.report["nbf"])
            ),
            "Expiry": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(self.report["exp"])
            ),
            "Issuer": self.report["iss"],
            "Remaining Time": time.strftime(
                "%H:%M:%S", time.gmtime(self.report["exp"] - int(time.time()))
            )
            if time.time() < self.report["exp"]
            else "Expired ❌",
            "Attestation Type": self.report["x-nvidia-attestation-type"],
            "Device ID": self.report["ueid"],
            "Secure Boot": (
                f"{self.status('secboot')} "
                f"{'Enabled' if self.report['secboot'] else 'Disabled'}"
            ),
            "Debugging": (
                f"{self.status('dbgstat')} "
                f"{'Enabled' if self.report['dbgstat'] == 'enabled' else 'Disabled'}"
            ),
            "HW Model": self.report["hwmodel"],
            "OEM ID": self.report["oemid"],
            "Driver Version": self.report["x-nvidia-gpu-driver-version"],
            "VBIOS Version": self.report["x-nvidia-gpu-vbios-version"],
        }
        return GPU_ATTESTATION_SUMMARY_TEMPLATE.format(**attestation_summary)
