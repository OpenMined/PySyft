data "google_project" "this" {
  project_id = var.project_id
}

resource "google_secret_manager_secret" "token" {
  secret_id = var.secret_name

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "token" {
  secret      = google_secret_manager_secret.token.id
  secret_data = file(var.token_file)
}

resource "google_secret_manager_secret_iam_member" "accessor" {
  secret_id = google_secret_manager_secret.token.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.enclave.email}"
}

locals {
  # The bootstrap dispatcher expects the project *number* form.
  secret_resource = "projects/${data.google_project.this.number}/secrets/${google_secret_manager_secret.token.secret_id}/versions/latest"
}
