#!/bin/bash

deploy() {
    # Params: $input_target $selected_cluster
    echo "Deploying to cluster "$2" on target "$1

    # 1. Make the Local cluster active
    kubectl config set current-context $2
    # 2. Create/update Cluster
    kustomize build overlays/$1/gateway/ | kubectl apply -f -
    kustomize build overlays/$1/node/ | kubectl apply -f -
}

usage() {
cat << EOF
Usage: ./deploy.sh -t <local|test|prod> [-ch]
Deploy PyGrid Gateway, Application Node to Kubernetes.

-h, Display help
-t, Specify the target environment (local|test|prod) to be deployed to
-c, Speicfy the cluster name

EOF
}

function in_array {
  ARRAY=$2
  for e in ${ARRAY[*]}
  do
    if [[ "$e" == "$1" ]]
    then
      return 0
    fi
  done
  return 1
}

# Array of supported targets
valid_targets=(local prod test)

input_target="local"
cluster_name=""

while getopts ":ht:c:" arg
do
    case $arg in
        t)
            input_target=${OPTARG}
            ;;
        c)
            cluster_name=${OPTARG}
            ;;
        h | *)
            usage
            exit 0
            ;;
        \? )
            usage
            exit 0
            ;;
    esac
done

if in_array $input_target "${valid_targets[*]}"
then
    if [[ $input_target == "local" && $cluster_name != "minikube" ]]
    then
        cluster_name="minikube"
        echo 'Only minikube is supported on local. Going ahead with the same'
    fi
    if [[ $input_target == "prod" || $input_target == "test"  ]]
    then
        if [[ -z "$cluster_name" ]]
        then
            usage
            exit 0
        fi
    fi
    deploy $input_target $cluster_name
else
    usage
    exit 0
fi