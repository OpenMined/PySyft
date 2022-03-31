variable "appId" {
  type =  string
  default = ""
  sensitive = true
}

variable "displayName" {}

variable "password" {
  type =  string
  default = ""
  sensitive = true
}

variable "tenant" {
  type =  string
  default = ""
  sensitive = true
}

variable "subscription_id" {
  type =  string
  default = ""
  sensitive = true
}

source "azure-arm" "domain" {
  azure_tags = {
    node_type = "domain"
    os_version = "ubuntu2004"
  }
  client_id                         = "${var.appId}"
  client_secret                     = "${var.password}"
  image_publisher                   = "canonical"
  image_offer                       = "0001-com-ubuntu-server-focal"
  image_version                     = "latest"
  image_sku                         = "20_04-lts"
  location                          = "West US"
  os_type                           = "Linux"
  subscription_id                   = "${var.subscription_id}"
  tenant_id                         = "${var.tenant}"
  vm_size                           = "Standard_D4s_v3"
  os_disk_size_gb                   = 128
  # vhd
  # capture_container_name            = "openmined-domain-ubuntu2004" # vhd
  # capture_name_prefix               = "openmined-domain" # vhd
  # resource_group_name               = "openmined-images" # vhd
  # storage_account                   = "openminedimgs" # vhd
  # managed image
  managed_image_name                = "openmined-domain-ubuntu2004-4" # managed image
  managed_image_resource_group_name = "openmined-images" # managed image
}

build {
  name = "openmined.node.domain"
  sources = ["source.azure-arm.domain"]

  provisioner "ansible" {
    playbook_file = "../ansible/site.yml"
    extra_arguments = [ "-v", "-e", "packer=true", "-e", "repo_branch=0.7.0" ]
  }

  provisioner "shell" {
    expect_disconnect = true
    scripts           = ["${path.root}/scripts/update.sh", "${path.root}/scripts/motd.sh", "${path.root}/scripts/hyperv.sh", "${path.root}/scripts/cleanup.sh"]
  }
}
