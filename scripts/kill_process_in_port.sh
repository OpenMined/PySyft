#!/bin/bash
lsof -i :$1| grep ":$1" | awk '{print $2}'| xargs -I {} bash -c 'kill -9 {}'