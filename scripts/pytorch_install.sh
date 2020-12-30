#!/bin/bash
set -e
TORCH_VERSION=$1

if [ $TORCH_VERSION = "1.4.0" ]
then
    TORCHVISION_VERSION="0.5.0"
elif [ $TORCH_VERSION = "1.5.0" ]
then
    TORCHVISION_VERSION="0.6.0"
elif [ $TORCH_VERSION = "1.5.1" ]
then
    TORCHVISION_VERSION="0.6.1"
elif [ $TORCH_VERSION = "1.6.0" ]
then
    TORCHVISION_VERSION="0.7"
    TORCHCSPRNG_VERSION="0.1.2"
elif [ $TORCH_VERSION = "1.7.0" ]
then
    TORCHVISION_VERSION="0.8.1"
    TORCHCSPRNG_VERSION="0.1.3"
elif [ $TORCH_VERSION = "1.7.1" ]
then
    TORCHVISION_VERSION="0.8.2"
    TORCHCSPRNG_VERSION="0.1.4"
fi
pip install torch==${TORCH_VERSION}
pip install torchvision==${TORCHVISION_VERSION}

# torchcsprng
if [ $TORCH_VERSION = "1.4.0" ]
then
    echo "No torchcsprng"
elif [ $TORCH_VERSION = "1.5.0" ]
then
    echo "No torchcsprng"
elif [ $TORCH_VERSION = "1.5.1" ]
then
    echo "No torchcsprng"
else
    pip install torchcsprng==${TORCHCSPRNG_VERSION}
fi

# check for error return codes
error=0
for pid in ${pids[*]}; do
    if ! wait $pid; then
        error=1
    fi
done

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

if [ "$error" -eq "0" ]; then
    printf "\n> PyTorch Install ${GREEN}PASSED${NC}\n"
else
    printf "\n> PyTorch Install ${RED}FAILED${NC}\n"
fi;

exit $error
