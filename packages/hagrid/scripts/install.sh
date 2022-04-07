#!/bin/sh

set -e

cat /dev/null <<EOF
------------------------------------------------------------------------
https://github.com/client9/shlib - portable posix shell functions
Public domain - http://unlicense.org
https://github.com/client9/shlib/blob/master/LICENSE.md
but credit (and pull requests) appreciated.
------------------------------------------------------------------------
EOF

is_command() {
  command -v "$1" >/dev/null
}

check_ubuntu() {
    if [ -f /etc/os-release ]
    then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ]
        then
            echo "$PRETTY_NAME detected"
        else
            echo "Install script currently only supports Ubuntu"
            exit 1
        fi
    else
        echo "Install script currently only supports Ubuntu"
        exit 1
    fi
}

apt_install() {
    sudo apt-get -qq -o=Dpkg::Use-Pty=0 update -y > /dev/null
    sudo apt-get -qq -o=Dpkg::Use-Pty=0 install $1 -y > /dev/null
}

hagrid_install() {
    pip install --quiet hagrid ansible ansible-core
}

check_and_install() {
    is_command $1
    BINARY_EXISTS=$?
    if [ "$BINARY_EXISTS" != "0" ]
    then
        echo "Installing missing dependency $2"
        sudo -n true
        HAS_SUDO=$?
        if [ "$HAS_SUDO" != "0" ]
        then
            echo "Installing missing dependency requires sudo"
            exit 1
        fi
        apt_install $2
        . ~/.profile
    fi
}

spinner="
🧙🏿‍♀️oo
o🧙‍♂️o
oo🧙🏽
"

spin() {
    while [ 1 ]
    do
        for i in $spinner do
        do
            clear && printf "\rHAGrid is installing:\n$i"
            sleep 0.2
        done
    done
}

execute() {
    check_ubuntu
    spin &
    pid=$!
    set +e
    check_and_install python3 python3
    check_and_install pip python3-pip
    check_and_install git git
    . ~/.profile
    hagrid_install
    hagrid >/dev/null
    kill $pid
    clear
    is_command hagrid
    BINARY_EXISTS=$?
    set -e
    if [ "$BINARY_EXISTS" != "0" ]
    then
        echo "HAGrid failed to install. Please try again"
        exit 1
    fi
    echo "HAGrid is installed! 🧙‍♂️"
    exec $SHELL
}

execute
