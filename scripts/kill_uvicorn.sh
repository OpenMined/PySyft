#!/bin/bash
lsof -i :8010| sed -n '2 p' | cut -d' ' -f2| xargs -I {} bash -c 'kill -9 {}'