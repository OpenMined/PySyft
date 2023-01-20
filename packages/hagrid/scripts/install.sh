#!/bin/sh

# run with:
# curl https://raw.githubusercontent.com/OpenMined/PySyft/dev/packages/hagrid/scripts/install.sh | sh

set -e

cat /dev/null <<EOF
------------------------------------------------------------------------
https://github.com/client9/shlib - portable posix shell functions
Public domain - http://unlicense.org
https://github.com/client9/shlib/blob/master/LICENSE.md
but credit (and pull requests) appreciated.
------------------------------------------------------------------------
EOF

OS_NOT_SUPPORTED="Install script currently only supports macOS & Ubuntu"

is_command() {
    set +e
    command -v "$1" >/dev/null
    RETURN_CODE=$?
    set +e
    return $RETURN_CODE
}

check_ubuntu() {
    if [ -f /etc/os-release ]
    then
        . /etc/os-release
        if [ "$ID" = "ubuntu" ]
        then
            return 0
        fi
    fi
    return 1
}

check_macos() {
    if [ "$(uname)" = "Darwin" ]
    then
        return 0
    fi
    return 1
}

check_os_supported() {
    echo "Checking OS..."
    if check_macos
    then
        echo "✅ macOS detected"
        return 0
    elif check_ubuntu
    then
        echo "✅ Ubuntu detected"
        return 0
    fi
    echo $OS_NOT_SUPPORTED
    exit 1
}

apt_install() {
    execute_sudo "apt-get -qq -o=Dpkg::Use-Pty=0 update -y"
    execute_sudo "apt-get -qq -o=Dpkg::Use-Pty=0 install $1 -y"
}

brew_install() {
    if is_command "brew"
    then
        # echo "Would run: brew install $1"
        HOMEBREW_NO_AUTO_UPDATE=1 brew install $1
    else
        echo "\nWe require brew to install packages.\nYou must install brew first: https://brew.sh/"
        exit 1
    fi
}

hagrid_install() {
    echo "\nChecking hagrid ..."
    if is_command "hagrid"
    then
        echo "✅ hagrid detected"
    else
        echo "Installing hagrid"
        pip install --quiet -U hagrid
    fi
}

# from: https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
unset HAVE_SUDO_ACCESS # unset this from the environment

abort() {
    printf "%s\n" "$@" >&2
    exit 1
}


have_sudo_access() {
    if [[ ! -x "/usr/bin/sudo" ]]
    then
        return 1
    fi

    local -a SUDO=("/usr/bin/sudo")
    if [[ -n "${SUDO_ASKPASS-}" ]]
    then
        SUDO+=("-A")
    elif [[ -n "${NONINTERACTIVE-}" ]]
    then
        SUDO+=("-n")
    fi

    if [[ -z "${HAVE_SUDO_ACCESS-}" ]]
    then
        if [[ -n "${NONINTERACTIVE-}" ]]
        then
            "${SUDO[@]}" -l mkdir &>/dev/null
        else
            "${SUDO[@]}" -v && "${SUDO[@]}" -l mkdir &>/dev/null
        fi
        HAVE_SUDO_ACCESS="$?"
    fi

    if [[ -n "${HOMEBREW_ON_MACOS-}" ]] && [[ "${HAVE_SUDO_ACCESS}" -ne 0 ]]
    then
        abort "Need sudo access on macOS (e.g. the user ${USER} needs to be an Administrator)!"
    fi

    return "${HAVE_SUDO_ACCESS}"
}

execute() {
    if ! "$@"
    then
        abort "$(printf "Failed during: %s" "$(shell_join "$@")")"
    fi
}

execute_sudo() {
    local -a args=("$@")
    if have_sudo_access
    then
        if [[ -n "${SUDO_ASKPASS-}" ]]
        then
            args=("-A" "${args[@]}")
        fi
        echo "/usr/bin/sudo" "${args[@]}"
        execute "/usr/bin/sudo" "${args[@]}"
    else
        echo "${args[@]}"
        execute "${args[@]}"
    fi
}

check_and_install() {
    echo "\nChecking $1 ..."
    if is_command $1
    then
        echo "✅ $1 detected"
        return 0
    else
        echo "Installing missing dependency $2"
        if check_macos
        then
            brew_install $2
        elif check_ubuntu
        then
            apt_install $2
        fi
    fi

    if is_command $1
    then
        echo "✅ $1 detected"
        return 0
    else
        echo "Failed to install $1. Please manually install it."
    fi
}

check_install_python() {
    check_and_install python3 python3
}

check_install_pip() {
    echo "\nChecking pip ..."
    if is_command "pip"
    then
        echo "✅ pip detected"
        return 0
    else
        if check_macos
        then
            echo "Installing missing dependency pip"
            python3 -m ensurepip
        else
            check_and_install pip python3-pip
        fi
    fi

    if is_command "pip"
    then
        echo "✅ pip detected"
    else
        echo "Failed to install pip. Please manually install it."
    fi
}

check_install_git() {
    check_and_install git git
}

execute() {
    check_os_supported
    check_install_python
    check_install_pip
    check_install_git

    hagrid_install

    if is_command "hagrid"
    then
        echo "\n🧙‍♂️ HAGrid is installed!\n"
        echo "To get started run: \n$ hagrid quickstart\n"
    else
        echo "\nHAGrid failed to install. Please try manually with:"
        echo "pip install -U hagrid"
        exit 1
    fi
}

execute
