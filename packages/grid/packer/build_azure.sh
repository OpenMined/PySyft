#!/bin/bash
packer init azure.pkr.hcl
PACKER_LOG=1 PACKER_LOG_PATH=./packer.log packer build -var-file=azure_vars.json -on-error=ask -var "subscription_id=${1}" azure.pkr.hcl
