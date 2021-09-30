#!/bin/sh -eux

msg='
This system is built by OpenMined.
More information can be found at https://github.com/OpenMined/PySyft'

if [ -d /etc/update-motd.d ]; then
    MOTD_CONFIG='/etc/update-motd.d/99-grid'

sudo su -c "cat >> $MOTD_CONFIG <<MSG
#!/bin/sh

cat <<'EOF'
$msg
EOF
MSG"

    sudo chmod 0755 "$MOTD_CONFIG"
else
    sudo su -c 'echo "$msg" >> /etc/motd'
fi
