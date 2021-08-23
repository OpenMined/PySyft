#!/bin/bash
packer init base.pkr.hcl
packer init domain.pkr.hcl
PACKER_LOG=1 PACKER_LOG_PATH=./packer.log packer build -on-error=ask -only='openmined.node.base.virtualbox-iso.ubuntu2004' base.pkr.hcl
PACKER_LOG=1 PACKER_LOG_PATH=./packer.log packer build -on-error=ask -only='openmined.node.domain.virtualbox-ovf.domain' domain.pkr.hcl