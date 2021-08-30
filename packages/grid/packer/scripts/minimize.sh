#!/bin/sh -eux

case "$PACKER_BUILDER_TYPE" in
    qemu) exit 0 ;;
esac

# Whiteout root
count=$(df --sync -kP / | tail -n1  | awk -F ' ' '{print $4}')
count=$(($count-1))
sudo dd if=/dev/zero of=/tmp/whitespace bs=1M count=$count || echo "dd exit code $? is suppressed";
sudo rm /tmp/whitespace

# Whiteout /boot
count=$(df --sync -kP /boot | tail -n1 | awk -F ' ' '{print $4}')
count=$(($count-1))
sudo dd if=/dev/zero of=/boot/whitespace bs=1M count=$count || echo "dd exit code $? is suppressed";
sudo rm /boot/whitespace

set +e
swapuuid="`/sbin/blkid -o value -l -s UUID -t TYPE=swap`";
case "$?" in
    2|0) ;;
    *) exit 1 ;;
esac
set -e

if [ "x${swapuuid}" != "x" ]; then
    # Whiteout the swap partition to reduce box size
    # Swap is disabled till reboot
    sudo swappart="`readlink -f /dev/disk/by-uuid/$swapuuid`";
    sudo /sbin/swapoff "$swappart";
    sudo dd if=/dev/zero of="$swappart" bs=1M || echo "dd exit code $? is suppressed";
    sudo /sbin/mkswap -U "$swapuuid" "$swappart";
fi

sudo sync;
