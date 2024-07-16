ATTESTATION_SERVICE_URL = (
    "http://localhost:4455"  # Replace with "http://attestation:4455"
)
ATTEST_CPU_ENDPOINT = "/attest/cpu"
ATTEST_GPU_ENDPOINT = "/attest/gpu"

CPU_ATTESTATION_SUMMARY_TEMPLATE = """
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
VM ID: {VM ID}

🔒 Security Features
-----------------------------------------------------------
Secure Boot: {Secure Boot}
Boot Debugging: {Boot Debugging}
Debuggers Disabled: {Debuggers Disabled}
Kernel Debugging: {Kernel Debugging}
Hypervisor Debugging: {Hypervisor Debugging}
Signing Disabled: {Signing Disabled}
Test Signing: {Test Signing}

💻 Operating System
-----------------------------------------------------------
OS Type: {OS Type}
OS Distro: {OS Distro}
OS Version: {OS Version}

🛡️ Compliance and Validation
-----------------------------------------------------------
PCRs Attested: {PCRs Attested}
DB Validation: {DB Validation}
DBX Validation: {DBX Validation}
Default Secure Boot Keys Validated: {Default Secure Boot Keys Validated}
Compliance Status: {Compliance Status}

🔐 Isolation Environment
-----------------------------------------------------------
Isolation Type: {Isolation Type}
Author Key Digest: {Author Key Digest}
Launch Measurement: {Launch Measurement}
Debuggability: {Debuggability}
Migration Allowed: {Migration Allowed}
-----------------------------------------------------------
"""
