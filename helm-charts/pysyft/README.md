# PySyft

## TL;DR

```console
$ helm dependency update
$ helm install pysyft .
```

## Introduction

This chart bootstraps a [PostgreSQL](https://artifacthub.io/packages/helm/bitnami/postgresql), [rabbitmq](https://artifacthub.io/packages/helm/bitnami/rabbitmq) and other dependent services on a [Kubernetes](http://kubernetes.io) cluster using the [Helm](https://helm.sh) package manager.

## Prerequisites

- Kubernetes 1.12+
- Helm 3.1.0
- PV provisioner support in the underlying infrastructure

## Installing the Chart
To install the chart with the release name `pysyft`:

```console
$ helm install pysyft .
```

The command deploys PostgreSQL on the Kubernetes cluster in the default configuration. The [Parameters](#parameters) section lists the parameters that can be configured during installation.

> **Tip**: List all releases using `helm list`

## Uninstalling the Chart

To uninstall/delete the `pysyft` deployment:

```console
$ helm uninstall pysyft
```

The command removes all the Kubernetes components but PVC's associated with the chart and deletes the release.

To delete the PVC's associated with `pysyft`:

```console
$ kubectl delete pvc -l release=pysyft
```

> **Note**: Deleting the PVC's will delete all data of services as well. Please be cautious before doing it.

## Parameters

### Global parameters

| Name                                    | Description                                                                          | Value |
| --------------------------------------- | ------------------------------------------------------------------------------------ | ----- |
| `global.imageRegistry`                  | Global Docker image registry                                                         | `""`  |
| `global.imagePullSecrets`               | Global Docker registry secret names as an array                                      | `[]`  |
| `global.storageClass`                   | Global StorageClass for Persistent Volume(s)                                         | `""`  |
| `global.postgresql.enabled`             | c                                            | `[]`  |
| `global.rabbitmq.enabled`               | Global Enable rabbitmq dependency chart                                              | `""`  |

### PostgreSQL parameters

Get the actual values to configure postgreSQL from the official documentation [link](https://artifacthub.io/packages/helm/bitnami/postgresql). Add `postgresql` variable before any parameter added.  

> **example**: *postgresql.image.registry* if the value in the documentation is *image.registry*.

### RabitMQ parameters

Get the actual values to configure RabitMQ from the official documentation [link](https://artifacthub.io/packages/helm/bitnami/rabbitmq). Add `rabbitmq` variable before any parameter added.  

> **example**: *rabbitmq.image.registry* if the value in the documentation is *image.registry*.

### Frontend, Backend, CeleryWorker parameters

Use the below Parameters which are common for all the charts *frontend*, *backend*, *celeryworker*. You need to add the respective tool name along with the parameter value when passing the parameters using command-line option *--set*. If using a values.yaml file then add the respective parameters under the service name as given in *values.yaml* file.

| Name                                            | Description                                                                                                    | Value                       |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `image.registry`                                | frontend image registry                                                                                        | `docker.io`                 |
| `image.repository`                              | frontend image repository                                                                                      | `openmined/grid-frontend`   |
| `image.tag`                                     | frontend image tag (immutable tags are recommended)                                                            | `latest`                    |
| `image.pullPolicy`                              | frontend image pull policy                                                                                     | `IfNotPresent`              |
| `image.pullSecrets`                             | Specify image pull secrets                                                                                     | `[]`                        |
| `podSecurityContext.fsGroup`                    | Group ID for the pod                                                                                           | `1001`                      |
| `securityContext.runAsUser`                     | User ID for the container                                                                                      | `1001`                      |
| `serviceAccount.create`                         | Specifies whether a service account should be created.                                                         | `false`                     |
| `serviceAccount.name`                           | Name of an already existing service account. If not set and create is true, a name is automatically generated. | `""`                        |
| `serviceAccount.annotations`                    | Annotations to add to the service account.                                                                     | `{}`                        |
| `replicaCount`                                  | Initial number of pods to spin up.                                                                             | `1`                         |
| `autoscaling.enabled`                           | Enable autoscaling of pods.                                                                                    | `false`                     |
| `autoscaling.minReplicas`                       | Minimum number of replicas needed for a pod.                                                                   | `1`                         |
| `autoscaling.maxReplicas`                       | Maximum number of replicas can a replicaset spin.                                                              | `10`                        |
| `autoscaling.targetCPUUtilizationPercentage`    | CPU limit on reaching which new pod will get triggered.                                                        | `80`                        |
| `autoscaling.targetMemoryUtilizationPercentage` | Memory limit on reaching which new pod will get triggered.                                                     | `80`                        |
| `service.type`                                  | Kubernetes Service type                                                                                        | `ClusterIP`                 |
| `service.port`                                  | frontend port                                                                                                  | `80`                        |
| `service.nodePort`                              | Specify the nodePort value for the LoadBalancer and NodePort service types                                     | `""`                        |
| `persistence.enabled`                           | Enable persistence using PVC                                                                                   | `true`                      |
| `persistence.existingClaim`                     | Provide an existing `PersistentVolumeClaim`, the value is evaluated as a template.                             | `""`                        |
| `persistence.mountPath`                         | The path the volume will be mounted at, useful when working on development.                                    | `/app`                      |
| `persistence.subPath`                           | The subdirectory of the volume to mount to                                                                     | `""`                        |
| `persistence.storageClass`                      | PVC Storage Class for frontend volume                                                                          | `""`                        |
| `persistence.accessModes`                       | PVC Access Mode for frontend volume                                                                            | `[]`                        |
| `persistence.size`                              | PVC Storage Request for frontend volume                                                                        | `8Gi`                       |
| `persistence.annotations`                       | Annotations for the PVC                                                                                        | `{}`                        |
| `resources.requests.cpu`                        | The requested cpu resources for the container                                                                  | `""`                        |
| `resources.requests.memory`                     | The requested memory resources for the container                                                               | `""`                        |
| `resources.limits.cpu`                          | The maximum cpu resources for the container                                                                    | `""`                        |
| `resources.limits.memory`                       | The maximum memory resources for the container                                                                 | `""`                        |
| `ingress.enabled`                               | Set to true to enable ingress record generation.                                                               | `false`                     |
| `ingress.annotations`                           | List of ingress annatotations.                                                                                 | `{}`                        |
| `ingress.hosts`                                 | List of ingress hosts and their paths.                                                                         | `[]`                        |
| `ingress.tls`                                   | List of ingress hosts and Secret names                                                                         | `[]`                        |

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`. For example,

```console
$ helm install pysyft \
  --set postgresql.postgresqlPassword=secretpassword,postgresql.postgresqlDatabase=my-database .
```

Alternatively, a YAML file that specifies the values for the parameters can be provided while installing the chart. For example,

```console
$ helm install pysyft -f values.yaml .
```

> **Tip**: You can use the default [values.yaml](values.yaml)

## Create the openmined namespace
$ kubectl get namespaces

## Installation using tilt and minikube

For development purposes, we're going to use [Tilt](https://tilt.dev/).
Please install it, so we can continue.

After installing the above, you should be able to run the following:

**Powershell**

``` pwsh
minikube start
minikube docker-env | Invoke-Expression
tilt up
```

**Linux shell**

``` bash
minikube start
eval $(minikube docker-env)
tilt up
```