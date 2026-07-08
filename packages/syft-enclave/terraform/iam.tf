resource "google_service_account" "enclave" {
  account_id   = var.service_account_id
  display_name = "Syft enclave service account"
  depends_on   = [google_project_service.apis]
}

# Project-level roles the Confidential Space launcher needs from the attached SA:
# - confidentialcomputing.workloadUser -> REST verifier client (attestation)
# - logging.logWriter -> launcher startup + diagnostic logs to Cloud Logging
resource "google_project_iam_member" "enclave_sa" {
  for_each = toset([
    "roles/confidentialcomputing.workloadUser",
    "roles/logging.logWriter",
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.enclave.email}"
}

# IAM grants are eventually consistent; a VM booting right after a fresh
# apply loses the race (the Confidential Space launcher 403s on
# confidentialcomputing.locations.list and exits, leaving a dead VM).
# Give the grants time to propagate before the instance boots.
resource "time_sleep" "iam_propagation" {
  create_duration = "120s"

  triggers = {
    sa = google_service_account.enclave.email
  }

  depends_on = [google_project_iam_member.enclave_sa]
}
