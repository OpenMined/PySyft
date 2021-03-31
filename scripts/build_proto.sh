#!/bin/bash
CLEAN="src/syft/proto"
PYTHON_OUT="src/syft"
PROTO_IN="proto"

command -v protoc &> /dev/null
if [ $? -ne 0 ]; then
    echo -ne "Install protobuf\n"
    exit 1
fi

rm -rf "${CLEAN}"
find ${PROTO_IN} -name "*.proto" -print0 | xargs -0 protoc --python_out=${PYTHON_OUT} --experimental_allow_proto3_optional

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

black src/syft/proto
isort src/syft/proto
