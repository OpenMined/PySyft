#!/bin/sh -eux

# Delete all Linux headers
sudo dpkg --list \
| awk '{ print $2 }' \
| grep 'linux-headers' \
| xargs sudo apt-get -y purge;

# Remove specific Linux kernels, such as linux-image-3.11.0-15-generic but
# keeps the current kernel and does not touch the virtual packages,
# e.g. 'linux-image-generic', etc.
sudo dpkg --list \
| awk '{ print $2 }' \
| grep 'linux-image-.*-generic' \
| grep -v `uname -r` \
| xargs sudo apt-get -y purge;

# Delete Linux source
sudo dpkg --list \
| awk '{ print $2 }' \
| grep linux-source \
| xargs sudo apt-get -y purge;

# Delete development packages
sudo dpkg --list \
| awk '{ print $2 }' \
| grep -- '-dev$' \
| xargs sudo apt-get -y purge;

# delete docs packages
sudo dpkg --list \
| awk '{ print $2 }' \
| grep -- '-doc$' \
| xargs sudo apt-get -y purge;

# Delete X11 libraries
sudo apt-get -y purge libx11-data xauth libxmuu1 libxcb1 libx11-6 libxext6;

# Delete obsolete networking
sudo apt-get -y purge ppp pppconfig pppoeconf;

# Delete oddities
sudo apt-get -y purge popularity-contest installation-report command-not-found friendly-recovery bash-completion fonts-ubuntu-font-family-console laptop-detect;

# 19.10+ don't have this package so fail gracefully
sudo apt-get -y purge command-not-found-data || true;

# Exlude the files we don't need w/o uninstalling linux-firmware
echo "==> Setup dpkg excludes for linux-firmware"
sudo su -c 'cat <<_EOF_ | cat >> /etc/dpkg/dpkg.cfg.d/excludes
#OM-BEGIN
path-exclude=/lib/firmware/*
path-exclude=/usr/share/doc/linux-firmware/*
#OM-END
_EOF_'

# Delete the massive firmware packages
sudo rm -rf /lib/firmware/*
sudo rm -rf /usr/share/doc/linux-firmware/*

sudo apt-get -y autoremove;
sudo apt-get -y clean;

# Remove docs
sudo rm -rf /usr/share/doc/*

# Remove caches
sudo find /var/cache -type f -exec rm -rf {} \;

# truncate any logs that have built up during the install
sudo find /var/log -type f -exec truncate --size=0 {} \;

# Blank netplan machine-id (DUID) so machines get unique ID generated on boot.
sudo truncate -s 0 /etc/machine-id

# remove the contents of /tmp and /var/tmp
sudo rm -rf /tmp/* /var/tmp/*

# clear the history so our install isn't there
export HISTSIZE=0
sudo rm -f /root/.wget-hsts