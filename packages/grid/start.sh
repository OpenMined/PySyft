#!/bin/bash
docker-compose --file "docker-compose.yml" --project-name "grid" --project-directory "/Users/andrewliamtrask/Dropbox/Laboratory/openmined/PySyft/PySyft/packages/grid" down
docker-compose up
