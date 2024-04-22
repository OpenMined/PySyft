# GCS Deployment

## Resource Links
* [Autopilot Overview](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview)
* [Autopilot Resource Limits Defaults](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-resource-requests#compute-class-defaults)
* [AutoPilot Resource Limits Min/Max](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-resource-requests#min-max-requests)
* [Compute Classes](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-compute-classes)
* [Performance Pods](https://cloud.google.com/kubernetes-engine/docs/how-to/performance-pods)

## Setup

Helm `values.yaml` for high & low side deployments
- [`gcp.high.yaml`](./gcp.high.yaml)
- [`gcp.low.yaml`](./gcp.low.yaml)

Deployment on GKE with SeaweedFS sync to GCS requires:
1. A GCS bucket in the same project where the cluster will be deployed
    * `syft-bucket-high` (for high side deployment)
    * `syft-bucket-low` (for low side deployment)

2. An IAM service account with sufficient permissions to read/write/delete object to these buckets

## For Autopilot Cluster

Uncomment the `nodeSelector` to use a specific compute class + machine family.

Set resource limits in-line with the machine family

```yaml
  nodeSelector:
    cloud.google.com/compute-class: Performance
    cloud.google.com/machine-family: c3
    
  resources:
    requests:
      cpu: 2
      memory: "8Gi"
    limits:
      cpu: 4
      memory: "16Gi"
```
