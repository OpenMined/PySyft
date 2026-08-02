data "google_compute_image" "confidential_space" {
  family  = var.dev_mode ? "confidential-space-debug" : "confidential-space"
  project = "confidential-space-images"
}

locals {
  # Production always encrypts
  # Dev mode defaults to off but honors the use_encryption override
  # Prod ignores the override.
  use_encryption = var.dev_mode ? coalesce(var.use_encryption, false) : true

  # image_digest (when set) always wins over image_tag
  container_image = var.image_digest != "" ? "${var.image_repo}@${var.image_digest}" : "${var.image_repo}:${var.image_tag}"

  # Hardware profiles. gpu is the only GPU config Confidential Space supports:
  # a3-highgpu-1g (1x H100, Intel TDX), flex-start provisioning (no on-demand
  # exists for it) — runs max_run_duration, then VM+disk auto-delete.
  hw = {
    cpu = { machine_type = "n2d-standard-2", confidential_type = "SEV", min_cpu_platform = "AMD Milan" }
    gpu = { machine_type = "a3-highgpu-1g", confidential_type = "TDX", min_cpu_platform = null }
  }[var.hardware]

  is_gpu = var.hardware == "gpu"

  machine_type = coalesce(var.machine_type, local.hw.machine_type)

  tee_metadata = merge(
    {
      "tee-image-reference"                 = local.container_image
      "tee-restart-policy"                  = var.dev_mode ? "Always" : "Never"
      "tee-env-SYFT_ENCLAVE_EMAIL"          = var.enclave_email
      "tee-env-SYFT_ENCLAVE_DATA_OWNERS"    = join(",", var.data_owners)
      "tee-env-SYFT_ENCLAVE_REQUIRE_TEE"    = "true"
      "tee-env-SYFT_BOOTSTRAP"              = "sa"
      "tee-env-SYFT_BOOTSTRAP_SA_SECRET"    = local.secret_resource
      "tee-env-SYFT_ENCLAVE_USE_ENCRYPTION" = local.use_encryption ? "true" : "false"
    },
    var.dev_mode ? { "tee-container-log-redirect" = "true" } : {},
    local.is_gpu ? {
      "tee-install-gpu-driver"  = "true"
      "tee-env-LD_LIBRARY_PATH" = "/usr/local/nvidia/lib64"
    } : {},
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
  machine_type     = local.machine_type
  min_cpu_platform = local.hw.min_cpu_platform # null (gpu): a3 pins Sapphire Rapids itself

  confidential_instance_config {
    # enable_confidential_compute is an SEV-only field — must be unset for TDX
    enable_confidential_compute = local.hw.confidential_type == "SEV" ? true : null
    confidential_instance_type  = local.hw.confidential_type # Confidential Space supports SEV/TDX, not SEV_SNP
  }

  shielded_instance_config {
    enable_secure_boot          = true
    enable_vtpm                 = true
    enable_integrity_monitoring = true
  }

  scheduling {
    on_host_maintenance         = "TERMINATE"
    provisioning_model          = local.is_gpu ? "FLEX_START" : null
    automatic_restart           = local.is_gpu ? false : true
    instance_termination_action = local.is_gpu ? "DELETE" : null

    dynamic "max_run_duration" {
      for_each = local.is_gpu ? [1] : []
      content {
        seconds = var.max_run_duration_seconds
      }
    }
  }

  dynamic "reservation_affinity" {
    for_each = local.is_gpu ? [1] : []
    content {
      type = "NO_RESERVATION"
    }
  }

  # Flex-start queues for H100 capacity — give the create call room to wait
  # instead of failing at terraform's 20m default. No fallback: it waits or errors.
  timeouts {
    create = "2h"
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
