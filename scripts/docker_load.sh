#!/bin/bash

cd .docker-cache/openmined
ls | grep .tar.gz | xargs -L 1 -I {} tar -xf {}
ls | grep -v .tar.gz | xargs -L 1 -I {} docker load -i {}
ls | grep -v .tar.gz | xargs rm
