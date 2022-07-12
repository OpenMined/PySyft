#!/bin/bash
echo $1
echo $2
(docker logs "${1}" -f &) | grep -q "${2}" || true
