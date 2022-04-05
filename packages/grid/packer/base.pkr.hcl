
source "virtualbox-iso" "ubuntu2004" {
  boot_command           = [
    "<enter><enter><f6><esc><wait> ",
    "autoinstall ds=nocloud-net;seedfrom=http://{{ .HTTPIP }}:{{ .HTTPPort }}/", "<enter><wait>"
  ]
  boot_wait              = "5s"
  guest_os_type          = "ubuntu-64"
  http_directory         = "cloud-config"
  iso_checksum           = "file:http://no.releases.ubuntu.com/20.04.4/SHA256SUMS"
  iso_url                = "http://no.releases.ubuntu.com/20.04.4/ubuntu-20.04.4-live-server-amd64.iso"
  memory                 = 4096
  disk_size              = 64000
  output_directory       = "output-base"
  output_filename        = "openmined.base.ubuntu2004"
  shutdown_command       = "sudo shutdown -P now"
  ssh_handshake_attempts = "100"
  ssh_password           = "ubuntu"
  ssh_pty                = true
  ssh_timeout            = "20m"
  ssh_username           = "ubuntu"
  host_port_min          = 2222
  host_port_max          = 2222
}

build {
  name = "openmined.node.base"
  sources = ["source.virtualbox-iso.ubuntu2004"]

  provisioner "shell" {
    inline = ["echo initial provisioning"]
  }

  post-processor "manifest" {
    output = "base-manifest.json"
  }
}
