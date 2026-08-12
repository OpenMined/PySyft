terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      # >= 7.41 for FLEX_START on google_compute_instance (gpu hardware profile)
      source  = "hashicorp/google"
      version = "~> 7.41"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.12"
    }
  }
}

provider "google" {
  project = var.project_id
  zone    = var.zone
}
