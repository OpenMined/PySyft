#!/usr/bin/env bash
echo Building image...
docker build --tag pysyft .
echo Build done!
if [ $(docker ps -a | grep My_PySyft | wc -l) -ge 1 ]; then
    echo A PySyft container is already running! Stopping it...
    docker stop  My_PySyft
    echo PySyft container stopped!
fi
echo Running PySyft container with name My_PySyft...
docker run -d --rm -p 8888:8888 --volume=$PWD/examples:/notebooks --name My_PySyft pysyft
if [ $(docker ps -a | grep My_PySyft | wc -l) -ge 1 ]; then
    echo PySyft is running! Open http://localhost:8888/ in your web browser
else
    echo Something went wrong.
fi
