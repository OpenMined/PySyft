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
