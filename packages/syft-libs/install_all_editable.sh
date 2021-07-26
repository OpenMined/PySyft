#!/bin/bash
ls | grep ^syft- | xargs -I{} bash -c "cd {}; pip install -e .; cd -"
