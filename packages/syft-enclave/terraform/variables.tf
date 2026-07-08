variable "project_id" {
  type        = string
  description = "GCP project ID (the ID, not the display name)."
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
  description = "GCP zone for the enclave VM."
}

variable "enclave_email" {
  type        = string
  description = "Email of the enclave datasite (SYFT_ENCLAVE_EMAIL)."
  validation {
    condition     = can(regex("@", var.enclave_email))
    error_message = "enclave_email must be an email address."
  }
}

variable "data_owners" {
  type        = list(string)
  description = "Emails of the data owners whose approval gates every job on this enclave."
  validation {
    condition     = length(var.data_owners) > 0 && alltrue([for e in var.data_owners : can(regex("@", e))])
    error_message = "data_owners must be a non-empty list of email addresses."
  }
}

variable "token_file" {
  type        = string
  description = "Path to the enclave's Google Drive token JSON. Uploaded to Secret Manager; the content also lands in the local terraform state file."
  validation {
    condition     = fileexists(var.token_file)
    error_message = "token_file does not exist."
  }
}

variable "vm_name" {
  type    = string
  default = "syft-enclave-vm"
}

variable "machine_type" {
  type    = string
  default = "n2d-standard-2"
}

variable "boot_disk_size_gb" {
  type    = number
  default = 200
}

variable "container_image" {
  type        = string
  default     = "docker.io/openminedreleasebot/syft-client-enclave:latest"
  description = "Enclave container image reference (tee-image-reference)."
}

variable "secret_name" {
  type    = string
  default = "syft-enclave-token"
}

variable "service_account_id" {
  type    = string
  default = "syft-enclave-service-account"
}

variable "dev_mode" {
  type        = bool
  default     = false
  description = "Debug bundle: confidential-space-debug image (SSH enabled), tee-restart-policy=Always, container logs to serial output, encryption off unless use_encryption overrides. Set via `just tf-apply` / `just tf-apply-dev`, not in tfvars."
}

variable "use_encryption" {
  type        = bool
  default     = null
  nullable    = true
  description = "Override SYFT_ENCLAVE_USE_ENCRYPTION. Default: true in production, false in dev mode."
}

variable "job_timeout_seconds" {
  type        = number
  default     = null
  nullable    = true
  description = "Optional SYFT_DEFAULT_JOB_TIMEOUT_SECONDS; unset means the container default (600s)."
}
