#!/bin/bash
CLEAN="src/syft/proto"
PYTHON_OUT="src/syft"
PROTO_IN="proto"

command -v protoc &> /dev/null
if [ $? -ne 0 ]; then
    echo -ne "Install protobuf (the 'protoc' tool: https://google.github.io/proto-lens/installing-protoc.html)\n"
    exit 1
fi

# check protoc version >= 3.15.0
VERSION=$(protoc --version | grep -o '[0-9].*')
# no easy way to do as bash does not support float comparisions.
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }
if version_lt $VERSION 3.15.0; then
    echo "you have Protobuf $VERSION, Please upgrade to Protobuf >= 3.15.0"
    exit 1
fi

echo "Protobuf Version: $(protoc --version)"
rm -rf "${CLEAN}"
find ${PROTO_IN} -name "*.proto" -print0 | xargs -0 protoc --python_out=${PYTHON_OUT}

# no easy way to do -i replace on both MacOS and Linux
# https://stackoverflow.com/questions/5694228/sed-in-place-flag-that-works-both-on-mac-bsd-and-linux
if [ "$(uname)" == "Darwin" ]; then
    echo "Darwin"
    # note the '' for empty file on MacOS
    find src/syft/proto -name "*_pb2.py" -print0 | xargs -0 sed -i '' 's/from \(proto.*\) import /from syft.\1 import /g'
else
    echo "Linux"
    find src/syft/proto -name "*_pb2.py" -print0 | xargs -0 sed -i 's/from \(proto.*\) import /from syft.\1 import /g'
fi
cd ../../ && isort . && cd - && black .
