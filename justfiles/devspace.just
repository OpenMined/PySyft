# ---------------------------------------------------------------------------------------------------------------------

[group('devspace')]
[private]
deploy-devspace cluster namespace *args='':
    #!/bin/bash
    set -euo pipefail

    cd packages/grid

    PROFILE="{{ _g_profiles }}"
    PROFILE=$(echo "$PROFILE" | sed -E 's/^,*|,*$//g')
    if [ -n "$PROFILE" ]; then
        PROFILE="-p $PROFILE"
    fi

    echo "Deploying to {{ cluster }}"

    devspace deploy -b \
        --no-warn \
        --kube-context {{ cluster }} \
        --namespace {{ namespace }} \
        $PROFILE \
        {{ args }} \
        --var CONTAINER_REGISTRY={{ registry_url }}

[group('devspace')]
[private]
purge-devspace cluster namespace:
    #!/bin/bash
    set -euo pipefail

    cd packages/grid
    devspace purge --force-purge --kube-context {{ cluster }} --no-warn --namespace {{ namespace }}
    sleep 3

# ---------------------------------------------------------------------------------------------------------------------