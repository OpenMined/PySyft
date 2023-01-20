#!/bin/bash
lsof -i :8010| grep "localhost:8010" | cut -d' ' -f2| xargs -I {} bash -c 'kill -9 {}'