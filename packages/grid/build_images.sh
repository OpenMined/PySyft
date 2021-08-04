#!/bin/bash
packer init domain.pkr.hcl
PACKER_LOG=1 PACKER_LOG_PATH=./packer.log packer build -on-error=ask domain.pkr.hcl
