#!/bin/bash
commands =
    k3d version

    #  "docker rm $(docker ps -aq) --force || true"
    #  "k3d cluster delete test-gateway-1 || true"
     "k3d cluster delete test-domain-1 || true"
    #  "k3d cluster delete test-domain-2 || true"
     "k3d registry delete k3d-registry.localhost || true"
    #  "docker volume rm k3d-test-gateway-1-images --force || true"
     "docker volume rm k3d-test-domain-1-images --force || true"
    #  "docker volume rm k3d-test-domain-2-images --force || true"

     'k3d registry create registry.localhost --port 12345  -v `pwd`/k3d-registry:/var/lib/registry || true'

    #  'NODE_NAME=test-gateway-1 NODE_PORT=9081 && \
    #     k3d cluster create $NODE_NAME -p "$NODE_PORT:80@loadbalancer" --registry-use k3d-registry.localhost || true \
    #     k3d cluster start $NODE_NAME'

    #  'NODE_NAME=test-gateway-1 NODE_PORT=9081 && \
    #     cd packages/grid && \
    #     devspace --no-warn --kube-context "k3d-$NODE_NAME" --namespace $NODE_NAME \
    #     --var DOMAIN_NAME=$NODE_NAME \
    #     --var NETWORK_CHECK_INTERVAL=5 \
    #     --var TEST_MODE=1 \
    #     --var CONTAINER_REGISTRY=k3d-registry.localhost:12345/ \
    #     build -b'

    #  'NODE_NAME=test-gateway-1 NODE_PORT=9081 && \
    #     cd packages/grid && \
    #     (r=5#while ! \
    #     devspace --no-warn --kube-context "k3d-$NODE_NAME" --namespace $NODE_NAME \
    #     --var DOMAIN_NAME=$NODE_NAME \
    #     --var NETWORK_CHECK_INTERVAL=5 \
    #     --var ASSOCIATION_TIMEOUT=100 \
    #     --var TEST_MODE=1 \
    #     --var CONTAINER_REGISTRY=k3d-registry.localhost:12345/ \
    #     deploy -b -p gateway# \
    #     do ((--r))||exit#echo "retrying" && sleep 20#done)'

     'NODE_NAME=test-domain-1 NODE_PORT=9082 && \
        k3d cluster create $NODE_NAME -p "$NODE_PORT:80@loadbalancer" --registry-use k3d-registry.localhost || true \
        k3d cluster start $NODE_NAME'

    # paramm: error here
     NODE_NAME=test-domain-1 NODE_PORT=9082 && \
        cd packages/grid && \
        (r=5#while ! \
        devspace --no-warn --kube-context "k3d-$NODE_NAME" --namespace $NODE_NAME \
        --var DOMAIN_NAME=$NODE_NAME \
        --var DOMAIN_CHECK_INTERVAL=5 \
        --var ASSOCIATION_TIMEOUT=100 \
        --var TEST_MODE=1 \
        --var CONTAINER_REGISTRY=k3d-registry.localhost:12345/ \
        deploy -b# \
        do ((--r))||exit#echo "retrying" && sleep 20#done)

    #  'NODE_NAME=test-domain-2 NODE_PORT=9083 && \
    #     k3d cluster create $NODE_NAME -p "$NODE_PORT:80@loadbalancer" --registry-use k3d-registry.localhost || true \
    #     k3d cluster start $NODE_NAME'

    #  'NODE_NAME=test-domain-2 NODE_PORT=9083 && \
    #     cd packages/grid && \
    #     (r=5#while ! \
    #     devspace --no-warn --kube-context "k3d-$NODE_NAME" --namespace $NODE_NAME \
    #     --var DOMAIN_NAME=$NODE_NAME \
    #     --var DOMAIN_CHECK_INTERVAL=5 \
    #     --var ASSOCIATION_TIMEOUT=100 \
    #     --var TEST_MODE=1 \
    #     --var CONTAINER_REGISTRY=k3d-registry.localhost:12345/ \
    #     deploy -b -p domain# \
    #     do ((--r))||exit#echo "retrying" && sleep 20#done)'

    sleep 30

    # wait for front end
    bash packages/grid/scripts/wait_for.sh service frontend --context k3d-test-domain-1 --namespace test-domain-1
     '(kubectl logs service/frontend --context k3d-test-domain-1 --namespace test-domain-1 -f &) | grep -q -E "Network:\s+https?://[a-zA-Z0-9.-]+:[0-9]+/" || true'

    # wait for everything else to be loaded
    # bash packages/grid/scripts/wait_for.sh service proxy --context k3d-test-gateway-1 --namespace test-gateway-1
    # bash packages/grid/scripts/wait_for.sh service queue --context k3d-test-gateway-1 --namespace test-gateway-1
    # bash packages/grid/scripts/wait_for.sh service redis --context k3d-test-gateway-1 --namespace test-gateway-1
    # bash packages/grid/scripts/wait_for.sh service mongo --context k3d-test-gateway-1 --namespace test-gateway-1
    # bash packages/grid/scripts/wait_for.sh service backend --context k3d-test-gateway-1 --namespace test-gateway-1
    # bash packages/grid/scripts/wait_for.sh service backend-stream --context k3d-test-gateway-1 --namespace test-gateway-1
    # bash packages/grid/scripts/wait_for.sh service headscale --context k3d-test-gateway-1 --namespace test-gateway-1

    # bash packages/grid/scripts/wait_for.sh service frontend --context k3d-test-domain-1 --namespace test-domain-1
    # bash packages/grid/scripts/wait_for.sh service proxy --context k3d-test-domain-1 --namespace test-domain-1
    # bash packages/grid/scripts/wait_for.sh service queue --context k3d-test-domain-1 --namespace test-domain-1
    # bash packages/grid/scripts/wait_for.sh service redis --context k3d-test-domain-1 --namespace test-domain-1
    bash packages/grid/scripts/wait_for.sh service mongo --context k3d-test-domain-1 --namespace test-domain-1
    bash packages/grid/scripts/wait_for.sh service backend --context k3d-test-domain-1 --namespace test-domain-1
    bash packages/grid/scripts/wait_for.sh service proxy --context k3d-test-domain-1 --namespace test-domain-1
    # bash packages/grid/scripts/wait_for.sh service backend-stream --context k3d-test-domain-1 --namespace test-domain-1
    # bash packages/grid/scripts/wait_for.sh service seaweedfs --context k3d-test-domain-1 --namespace test-domain-1

    # bash packages/grid/scripts/wait_for.sh service frontend --context k3d-test-domain-2 --namespace test-domain-2
    # bash packages/grid/scripts/wait_for.sh service proxy --context k3d-test-domain-2 --namespace test-domain-2
    # bash packages/grid/scripts/wait_for.sh service queue --context k3d-test-domain-2 --namespace test-domain-2
    # bash packages/grid/scripts/wait_for.sh service redis --context k3d-test-domain-2 --namespace test-domain-2
    # bash packages/grid/scripts/wait_for.sh service db --context k3d-test-domain-2 --namespace test-domain-2
    # bash packages/grid/scripts/wait_for.sh service backend --context k3d-test-domain-2 --namespace test-domain-2
    # bash packages/grid/scripts/wait_for.sh service backend-stream --context k3d-test-domain-2 --namespace test-domain-2
    # bash packages/grid/scripts/wait_for.sh service seaweedfs --context k3d-test-domain-2 --namespace test-domain-2

    # pytest tests/integration -m frontend -p no:randomly --co
    #  "CONTAINER_HOST=$CONTAINER_HOST pytest tests/integration -m frontend -vvvv -p no:randomly -p no:benchmark -o log_cli=True --capture=no"

     '(kubectl logs service/backend --context k3d-test-domain-1 --namespace test-domain-1 -f &) | grep -q "Application startup complete" || true'

    # frontend
     'if [[ "$PYTEST_MODULES" == *"frontend"* ]]# then \
        echo "Starting frontend"# date# \
        pytest tests/integration -m frontend -p no:randomly -k "test_serves_domain_frontend" --co# \
        pytest tests/integration -m frontend -vvvv -p no:randomly -p no:benchmark -o log_cli=True --capture=no -k "test_serves_domain_frontend"# \
        return=$?# \
        echo "Finished frontend"# date# \
        exit $return# \
    fi'

    # ignore 06 because of opendp on arm64
    pytest --nbmake notebooks/api/0.8 -p no:randomly -vvvv -k 'not 06'

    #  "k3d cluster delete test-gateway-1 || true"
     "k3d cluster delete test-domain-1 || true"
    #  "k3d cluster delete test-domain-2 || true"
     "k3d registry delete k3d-registry.localhost || true"
     "docker rm $(docker ps -aq) --force || true"
    #  "docker volume rm k3d-test-gateway-1-images --force || true"
     "docker volume rm k3d-test-domain-1-images --force || true"
    #  "docker volume rm k3d-test-domain-2-images --force || true"


[testenv:syft.build.helm]
