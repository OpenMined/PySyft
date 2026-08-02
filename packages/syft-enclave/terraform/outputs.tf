output "vm_name" {
  value = google_compute_instance.enclave.name
}

output "zone" {
  value = var.zone
}

output "project_id" {
  value = var.project_id
}

output "service_account_email" {
  value = google_service_account.enclave.email
}

output "secret_resource" {
  value = local.secret_resource
}

output "container_image" {
  description = "Resolved image reference deployed as tee-image-reference."
  value       = local.container_image
}

output "hardware_config" {
  description = "Resolved hardware profile."
  value = {
    hardware           = var.hardware
    machine_type       = local.machine_type
    confidential_type  = local.hw.confidential_type
    provisioning_model = local.is_gpu ? "FLEX_START" : "STANDARD"
    max_run_duration   = local.is_gpu ? "${var.max_run_duration_seconds}s" : null
  }
}

output "vm_external_ip" {
  description = "Outbound-only: no inbound port is open on the enclave."
  value       = google_compute_instance.enclave.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  value = var.dev_mode ? "gcloud compute ssh ${var.vm_name} --project=${var.project_id} --zone=${var.zone}" : "(dev_mode only — the production image has no SSH)"
}