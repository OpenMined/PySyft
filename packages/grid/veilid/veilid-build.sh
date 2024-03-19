#!/bin/sh
DEBUG_BUILD=${DEBUG_BUILD:-"0"}
rm ./veilid-server
cd veilid/veilid-server
if [ $DEBUG_BUILD = "0" ]; then
    cargo build --release -p veilid-server
else
    cargo build -p veilid-server
fi

cd ../..
if [ $DEBUG_BUILD = "0" ]; then
    cp ./veilid/target/release/veilid-server ./
    echo "Finished Building Release"
else
    cp ./veilid/target/debug/veilid-server ./
    echo "Finished Building Debug"
fi