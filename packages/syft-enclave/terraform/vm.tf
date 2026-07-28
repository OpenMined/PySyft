data "google_compute_image" "confidential_space" {
  family  = var.dev_mode ? "confidential-space-debug" : "confidential-space"
  project = "confidential-space-images"
}

locals {
  # Production always encrypts
  # Dev mode defaults to off but honors the use_encryption override
  # Prod ignores the override.
  use_encryption = var.dev_mode ? coalesce(var.use_encryption, false) : true

  tee_metadata = merge(
    {
      "tee-image-reference"                 = var.container_image
      "tee-restart-policy"                  = var.dev_mode ? "Always" : "Never"
      "tee-env-SYFT_ENCLAVE_EMAIL"          = var.enclave_email
      "tee-env-SYFT_ENCLAVE_DATA_OWNERS"    = join(",", var.data_owners)
      "tee-env-SYFT_ENCLAVE_REQUIRE_TEE"    = "true"
      "tee-env-SYFT_BOOTSTRAP"              = "sa"
      "tee-env-SYFT_BOOTSTRAP_SA_SECRET"    = local.secret_resource
      "tee-env-SYFT_ENCLAVE_USE_ENCRYPTION" = local.use_encryption ? "true" : "false"
    },
    var.dev_mode ? { "tee-container-log-redirect" = "true" } : {},
    var.job_timeout_seconds != null
    ? { "tee-env-SYFT_DEFAULT_JOB_TIMEOUT_SECONDS" = tostring(var.job_timeout_seconds) }
    : {}
  )
}

# Confidential Space reads tee-* metadata only at boot; an in-place metadata
# update would silently do nothing. Force VM replacement on any tee change.
resource "terraform_data" "tee_metadata" {
  triggers_replace = local.tee_metadata
}

resource "google_compute_instance" "enclave" {
  name             = var.vm_name
  zone             = var.zone
  machine_type     = var.machine_type
  min_cpu_platform = "AMD Milan"

  confidential_instance_config {
    enable_confidential_compute = true
    confidential_instance_type  = "SEV" # Confidential Space supports SEV/TDX, not SEV_SNP
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  scheduling {
    on_host_maintenance = "MIGRATE" # allowed for SEV only
  }

  boot_disk {
    initialize_params {
      image = data.google_compute_image.confidential_space.self_link
      size  = var.boot_disk_size_gb
      type  = "pd-ssd"
    }
  }

  # No network tags and no firewall rules: nothing may open an inbound port
  # on the enclave. The ephemeral external IP is for outbound traffic only
  # (Google Drive polling, container pull).
  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    email  = google_service_account.enclave.email
    scopes = ["cloud-platform"]
  }

  metadata = local.tee_metadata

  lifecycle {
    replace_triggered_by = [terraform_data.tee_metadata]
  }

  # The launcher fetches the token and attests on first boot — everything it
  # touches must exist (and IAM must have propagated) before the VM starts.
  depends_on = [
    google_secret_manager_secret_version.token,
    google_secret_manager_secret_iam_member.accessor,
    time_sleep.iam_propagation,
  ]
}
