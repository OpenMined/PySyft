#!/bin/bash

usage() { echo "Usage: $0 [-g <gpu>] [-m <grid-mode>] [-a <host_ip>] [-n <name>] [-e <email>]" 1>&2; exit 1; }

# default is not gpu
gpu=0

while getopts ":gm:a:n:e:" o; do
    case "${o}" in
        g)
            gpu=1
            ;;
        m)
            mode=${OPTARG}
            [ "${mode}" == "compute" ] || [ "${mode}" == "tree" ] || [ "${mode}" == "anchor" ]|| usage
            ;;
        a)
            haddr=${OPTARG}
            [[ ${haddr} =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]] || usage
            ;;
        n)
            name=${OPTARG}
            ;;
        e)
            email=${OPTARG}
            regex="^[a-z0-9!#\$%&'*+/=?^_\`{|}~-]+(\.[a-z0-9!#$%&'*+/=?^_\`{|}~-]+)*@([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z0-9]([a-z0-9-]*[a-z0-9])?\$"
            [[ $email =~ $regex ]] || usage

            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${mode}" ] || [ -z "${haddr}" ] || [ -z "${name}" ] || [ -z "${email}" ] ; then
    usage
fi

# setup env variable read by the worker
export NAME="${name}"
export EMAIL="${email}"
export GRID_MODE="${mode}"
export IPFS_ADDR="${haddr}"


if [ $gpu -eq 0 ] ; then
    echo 'CPU mode'
    docker-compose build
    docker-compose up ; else
    echo 'GPU mode'
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml build
    docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
fi

