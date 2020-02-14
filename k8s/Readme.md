
# PyGrid on Kubernetes

Use `deploy.sh` to deploy to a local cluster, a test cluster or a prod cluster

```shell
  ./deploy.sh -t <local|test|prod> [-ch]

  Deploy PyGrid Gateway, Application Node to Kubernetes.

    -h, Display help
    -t, Specify the target environment (local|test|prod) to be deployed to
    -c, Specify the cluster name
```

## Running on local minikube setup

* Steps to run things locally
* Minikube

  ```shell
  $ minikube start
  $ eval $(minikube docker-env) (on bash)
  $ eval (minikube docker-env) (on fish)
  ```

* Build Docker images

  ```shell
  $ docker build -t openmined/grid-node ./app/websocket/  # Build PyGrid node image
  $ docker build -t openmined/grid-gateway ./gateway/  # Build PyGrid Gateway image
  $ ./deploy.sh -t local # Kubernetes local deployment
  ```

* When running locally, given that the services are exposed as LoadBalancer use the below command to get thigns running

  ```shell
    $ minikube tunnel

  ```

Ref: [Running Local Docker images in Kubernetes](https://dzone.com/articles/running-local-docker-images-in-kubernetes-1)


## Running on remote (test, prod)

* Create a cluster in GCP/EKS/AKS and update the same in deploy. Please do not commit the same to git
* Get the Kubeconfig from the respective cluster.
  * Stackoverflow [link](https://stackoverflow.com/questions/48394610/connect-local-instance-of-kubectl-to-gke-cluster-without-using-gcloud-tool) for GCP.
* Run deploy

  ```shell
    $ ./deploy.sh -t test -c <cluster_name>
  ```
