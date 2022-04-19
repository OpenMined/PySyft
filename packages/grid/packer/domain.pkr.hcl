source "virtualbox-ovf" "domain" {
  shutdown_command = "echo 'vagrant' | sudo -S shutdown -P now"
  source_path      = "output-base/openmined.base.ubuntu2004.ovf"
  ssh_password     = "ubuntu"
  ssh_port         = 22
  ssh_username     = "ubuntu"
  host_port_min    = 2222
  host_port_max    = 2222
  output_directory = "output-domain"
  output_filename  = "openmined.domain.ubuntu2004"
}

build {
  name = "openmined.node.domain"
  sources = ["source.virtualbox-ovf.domain"]

  provisioner "ansible" {
    playbook_file = "../ansible/site.yml"
    extra_arguments = [ "-v", "-e", "packer=true", "-e", "repo_branch=0.7.0" ]
  }

  provisioner "shell" {
    expect_disconnect = true
    scripts           = ["${path.root}/scripts/update.sh", "${path.root}/scripts/motd.sh", "${path.root}/scripts/hyperv.sh", "${path.root}/scripts/cleanup.sh", "${path.root}/scripts/minimize.sh"]
  }

  post-processor "vagrant" {
    keep_input_artifact  = true
    provider_override    = "virtualbox"
    output               = "output-domain/openmined.domain.ubuntu2004.box"
    vagrantfile_template = "Vagrantfile"
  }
}
