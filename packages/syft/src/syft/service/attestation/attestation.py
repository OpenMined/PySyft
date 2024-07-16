# stdlib
import time
from typing import Any

# relative
from .attestation_constants import CPU_ATTESTATION_SUMMARY_TEMPLATE


class AttestationReport:
    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.expected_values = {
            "secureboot": True,
            "x-ms-azurevm-bootdebug-enabled": False,
            "x-ms-azurevm-debuggersdisabled": True,
            "x-ms-azurevm-kerneldebug-enabled": False,
            "x-ms-azurevm-hypervisordebug-enabled": False,
            "x-ms-azurevm-signingdisabled": True,
            "x-ms-azurevm-testsigning-enabled": False,
            "x-ms-azurevm-dbvalidated": True,
            "x-ms-azurevm-dbxvalidated": True,
            "x-ms-azurevm-default-securebootkeysvalidated": True,
            "x-ms-isolation-tee.x-ms-sevsnpvm-is-debuggable": False,
            "x-ms-isolation-tee.x-ms-sevsnpvm-migration-allowed": False,
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
            ),
            "Attestation Type": self.report["x-ms-attestation-type"],
            "VM ID": self.report["x-ms-azurevm-vmid"],
            "Secure Boot": (
                f"{self.status('secureboot')} "
                f"{'Enabled' if self.report['secureboot'] else 'Disabled'}"
            ),
            "Boot Debugging": (
                f"{self.status('x-ms-azurevm-bootdebug-enabled')} "
                f"{'Enabled' if self.report['x-ms-azurevm-bootdebug-enabled'] else 'Disabled'}"
            ),
            "Debuggers Disabled": (
                f"{self.status('x-ms-azurevm-debuggersdisabled')} "
                f"{'Yes' if self.report['x-ms-azurevm-debuggersdisabled'] else 'No'}"
            ),
            "Kernel Debugging": (
                f"{self.status('x-ms-azurevm-kerneldebug-enabled')} "
                f"{'Enabled' if self.report['x-ms-azurevm-kerneldebug-enabled'] else 'Disabled'}"
            ),
            "Hypervisor Debugging": (
                f"{self.status('x-ms-azurevm-hypervisordebug-enabled')} "
                f"{'Enabled' if self.report['x-ms-azurevm-hypervisordebug-enabled'] else 'Disabled'}"
            ),
            "Signing Disabled": (
                f"{self.status('x-ms-azurevm-signingdisabled')} "
                f"{'Yes' if self.report['x-ms-azurevm-signingdisabled'] else 'No'}"
            ),
            "Test Signing": (
                f"{self.status('x-ms-azurevm-testsigning-enabled')} "
                f"{'Enabled' if self.report['x-ms-azurevm-testsigning-enabled'] else 'Disabled'}"
            ),
            "OS Type": self.report["x-ms-azurevm-ostype"].capitalize(),
            "OS Distro": self.report["x-ms-azurevm-osdistro"],
            "OS Version": (
                f"{self.report['x-ms-azurevm-osversion-major']}."
                f"{self.report['x-ms-azurevm-osversion-minor']}"
            ),
            "PCRs Attested": ", ".join(
                map(str, self.report["x-ms-azurevm-attested-pcrs"])
            ),
            "DB Validation": (
                f"{self.status('x-ms-azurevm-dbvalidated')} "
                f"{'Valid' if self.report['x-ms-azurevm-dbvalidated'] else 'Invalid'}"
            ),
            "DBX Validation": (
                f"{self.status('x-ms-azurevm-dbxvalidated')} "
                f"{'Valid' if self.report['x-ms-azurevm-dbxvalidated'] else 'Invalid'}"
            ),
            "Default Secure Boot Keys Validated": (
                f"{self.status('x-ms-azurevm-default-securebootkeysvalidated')} "
                f"{'Yes' if self.report['x-ms-azurevm-default-securebootkeysvalidated'] else 'No'}"
            ),
            "Compliance Status": (
                self.report["x-ms-isolation-tee"]["x-ms-compliance-status"]
                .replace("-", " ")
                .capitalize()
            ),
            "Isolation Type": (
                self.report["x-ms-isolation-tee"]["x-ms-attestation-type"].upper()
            ),
            "Author Key Digest": (
                self.report["x-ms-isolation-tee"]["x-ms-sevsnpvm-authorkeydigest"]
            ),
            "Launch Measurement": (
                self.report["x-ms-isolation-tee"]["x-ms-sevsnpvm-launchmeasurement"]
            ),
            "Debuggability": (
                f"{self.status('x-ms-isolation-tee.x-ms-sevsnpvm-is-debuggable')} "
                f"{'Yes' if self.report['x-ms-isolation-tee']['x-ms-sevsnpvm-is-debuggable'] else 'No'}"
            ),
            "Migration Allowed": (
                f"{self.status('x-ms-isolation-tee.x-ms-sevsnpvm-migration-allowed')} "
                f"{'Yes' if self.report['x-ms-isolation-tee']['x-ms-sevsnpvm-migration-allowed'] else 'No'}"
            ),
        }
        return CPU_ATTESTATION_SUMMARY_TEMPLATE.format(**attestation_summary)
