# disable_on_destroy = false: `terraform destroy` must not disable
# project-wide APIs that other workloads in the project may depend on.
resource "google_project_service" "apis" {
  for_each = toset([
    "compute.googleapis.com",
    "confidentialcomputing.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}
