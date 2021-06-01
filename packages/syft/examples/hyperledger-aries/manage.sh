#!/bin/bash

export MSYS_NO_PATHCONV=1
export DOCKERHOST=${APPLICATION_URL-$(docker run --rm --net=host codenvy/che-ip)}
set -e

S2I_EXE=s2i
if [ -z $(type -P "$S2I_EXE") ]; then
    echo -e "The ${S2I_EXE} executable is needed and not on your path."
    echo -e "It can be downloaded from here: https://github.com/openshift/source-to-image/releases"
    echo -e "Make sure you extract the binary and place it in a directory on your path."
    exit 1
fi

SCRIPT_HOME="$(cd "$(dirname "$0")" && pwd)"

# =================================================================================================================
# Usage:
# -----------------------------------------------------------------------------------------------------------------
usage() {
  cat <<-EOF

      Usage: $0 [command] [options]

      Commands:

      up -  Builds the images, creates the application containers
            and starts the services based on the docker-compose.yml file.

            You can pass in a list of containers to start.
            By default all containers will be started.

            The API_URL used by tob-web can also be redirected.

            Examples:
            $0 start
            $0 start

      start - Same as up

      restart - Re-starts the application containers,
                useful when updating one of the container images during development.

            You can pass in a list of containers to be restarted.
            By default all containers will be restarted.

            Examples:
            $0 start

      logs - Display the logs from the docker compose run (ctrl-c to exit).

      stop - Stops the services.  This is a non-destructive process.  The volumes and containers
             are not deleted so they will be reused the next time you run start.

      down - Brings down the services and removes the volumes (storage) and containers.
      rm - Same as down


EOF
    exit 1
}
# -----------------------------------------------------------------------------------------------------------------
# Default Settings:
# -----------------------------------------------------------------------------------------------------------------
DEFAULT_CONTAINERS="data-owner-wallet-db ngrok-data-owner data-owner-agent data-owner-business-logic
data-scientist-wallet-db ngrok-data-scientist data-scientist-agent data-scientist-business-logic
om-authority-wallet-db ngrok-om-authority om-authority-agent om-authority-business-logic"

# DEFAULT_CONTAINERS="data-owner-agent data-scientist-agent"
# -----------------------------------------------------------------------------------------------------------------
# Functions:
# -----------------------------------------------------------------------------------------------------------------
function echoRed() {
    _msg=${1}
    _red='\e[31m'
    _nc='\e[0m' # No Color
    echo -e "${_red}${_msg}${_nc}"
}

function echoYellow() {
    _msg=${1}
    _yellow='\e[33m'
    _nc='\e[0m' # No Color
    echo -e "${_yellow}${_msg}${_nc}"
}

configureEnvironment() {

    if [ -f .env ]; then
        while read line; do
            if [[ ! "$line" =~ ^\# ]] && [[ "$line" =~ .*= ]]; then
                export ${line//[$'\r\n']/}
            fi
        done <.env
    fi

    for arg in "$@"; do
        # Remove recognized arguments from the list after processing.
        shift

        # echo "arg: ${arg}"
        # echo "Remaining: ${@}"

        case "$arg" in
            *=*)
                # echo "Exporting ..."
                export "${arg}"
            ;;
            *)
                # echo "Saving for later ..."
                # If not recognized, save it for later procesing ...
                set -- "$@" "$arg"
            ;;
        esac
    done




}

getInputParams() {
    ARGS=""

    for arg in $@; do
        case "$arg" in
            *=*)
                # Skip it
            ;;
            *)
                ARGS+=" $arg"
            ;;
        esac
    done

    echo ${ARGS}
}

getProductionParams() {
    CONTAINERS=""
    ARGS=""

    for arg in $@; do
        case "$arg" in
            *=*)
                # Skip it
            ;;
            -*)
                ARGS+=" $arg"
            ;;
            *)
                CONTAINERS+=" $arg"
            ;;
        esac
    done

    if [ -z "$CONTAINERS" ]; then
        CONTAINERS="$PRODUCTION_CONTAINERS"
    fi

    echo ${ARGS} ${CONTAINERS}
}

getStartupParams() {
    CONTAINERS=""
    ARGS=""

    for arg in $@; do
        case "$arg" in
            *=*)
                # Skip it
            ;;
            -*)
                ARGS+=" $arg"
            ;;
            *)
                CONTAINERS+=" $arg"
            ;;
        esac
    done

    if [ -z "$CONTAINERS" ]; then
        CONTAINERS="$DEFAULT_CONTAINERS"
    fi

    echo ${ARGS} ${CONTAINERS}
}

deleteVolumes() {
    _projectName=${COMPOSE_PROJECT_NAME:-docker}

    echo "Stopping and removing any running containers ..."
    docker-compose down -v

    _pattern="^${_projectName}_\|^docker_"
    _volumes=$(docker volume ls -q | grep ${_pattern})

    if [ ! -z "${_volumes}" ]; then
        echo "Removing project volumes ..."
        echo ${_volumes} | xargs docker volume rm
    else
        echo "No project volumes exist."
    fi

    echo "Removing build cache ..."
    rm -Rf ../client/tob-web/.cache
}


getSeedJson() {
    _seed=${1}
    if [ -z "${_seed}" ]; then
        echo -e \\n"getSeedJson; Missing parameter!"\\n
        exit 1
    fi

    echo "{\"seed\": \"${_seed}\"}"
}

generateSeeds() {
    echo ${INDY_WALLET_SEED}
}



toLower() {
    echo $(echo ${@} | tr '[:upper:]' '[:lower:]')
}

echoError() {
    _msg=${1}
    _red='\033[0;31m'
    _nc='\033[0m' # No Color
    echo -e "${_red}${_msg}${_nc}" >&2
}

functionExists() {
    (
        if [ ! -z ${1} ] && type ${1} &>/dev/null; then
            return 0
        else
            return 1
        fi
    )
}
# =================================================================================================================

pushd "${SCRIPT_HOME}" >/dev/null
COMMAND=$(toLower ${1})
shift || COMMAND=usage

case "${COMMAND}" in
    start | up)
        echoYellow "Starting up... This can take a couple of minutes."
        _startupParams=$(getStartupParams $@)
        configureEnvironment "$@"
        docker-compose\
        --log-level ERROR up \
        --build --remove-orphans \
        -d ${_startupParams}
        docker-compose \
        --log-level ERROR logs \
        -f
    ;;
    production)
        echoYellow "Starting up... This can take a couple of minutes."
        _startupParams=$(getProductionParams $@)
        configureEnvironment "$@"
        docker-compose\
        -f docker-compose.prod.yml \
        --log-level ERROR up \
        --build --remove-orphans \
        -d ${_startupParams}
        docker-compose \
        --log-level ERROR logs \
        -f
    ;;
    restart)
        _startupParams=$(getStartupParams $@)
        configureEnvironment "$@"
        docker-compose stop ${_startupParams}
        docker-compose up -d --build --remove-orphans ${_startupParams}
    ;;
    logs)
        configureEnvironment "$@"
        docker-compose logs -f
    ;;
    stop)
        configureEnvironment
        docker-compose stop
    ;;
    rm | down)
        configureEnvironment
        docker-compose \
        --log-level ERROR down \
        -v
        usage
    ;;
esac

popd >/dev/null
