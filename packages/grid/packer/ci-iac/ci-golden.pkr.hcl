variable "appId" {
  type =  string
  default = ""
  sensitive = true
}


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

source "azure-arm" "ci" {
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
  vm_size                           = "Standard_D8ds_v5"
  os_disk_size_gb                   = 512
  managed_image_name                = "ubuntu-ci-managed-image"
  managed_image_resource_group_name = "CI-Images"

}

build {
  name = "openmined.golden-image.ci"
  sources = ["source.azure-arm.ci"]

  provisioner "ansible" {
    playbook_file = "./ansible/site.yml"
    extra_arguments = [ "-v", "-e", "packer=true", "-e", "repo_branch=0.7.0" ]
  }

  provisioner "shell" {
    expect_disconnect = true
    scripts           = ["${path.root}/scripts/update.sh", "${path.root}/scripts/motd.sh", "${path.root}/scripts/hyperv.sh", "${path.root}/scripts/cleanup.sh"]
  }

  provisioner "shell" {
    expect_disconnect = true
    script           = ["${path.root}/scripts/setup_githubrunner.sh"]
  }

  provisioner "shell" {
    execute_command = "chmod +x {{ .Path }}; {{ .Vars }} sudo -E sh '{{ .Path }}'"
    inline          = ["/usr/sbin/waagent -force -deprovision+user && export HISTSIZE=0 && sync"]
    inline_shebang  = "/bin/sh -x"
  }
}

