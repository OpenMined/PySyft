packer {
  required_plugins {
    amazon = {
      version = ">= 0.0.2"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

source "vagrant" "ubuntu2004" {
  communicator = "ssh"
  provider     = "virtualbox"
  source_path  = "ubuntu/focal64"
}

build {
  name = "openmined.domain"
  sources = ["source.vagrant.ubuntu2004"]

  provisioner "ansible" {
    playbook_file = "ansible/site.yml"
    override = {
      vagrant = {
        ubuntu2004 = {
          ansible_env_vars = ["packer=true"]
        }
      }
    }
    extra_arguments = [ "-v" ]
  }
}
