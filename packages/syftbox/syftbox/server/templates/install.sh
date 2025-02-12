#!/bin/sh

set -e

# --no-prompt => disables the run client prompt
ASK_RUN_CLIENT=1

# --run => disables the prompt & runs the client
RUN_CLIENT=0

# --system-python => MANAGED_PYTHON=0
# --managed-python => MANAGED_PYTHON=1
MANAGED_PYTHON=1
MANAGED_PYTHON_VERSION=${MANAGED_PYTHON_VERSION:-"3.12"}

# min system python version if not using managed python
REQ_PYTHON_MAJOR="3"
REQ_PYTHON_MINOR="9"

red='\033[1;31m'
yellow='\033[0;33m'
cyan='\033[0;36m'
green='\033[1;32m'
reset='\033[0m'

err() {
    echo "${red}ERROR${reset}: $1" >&2
    exit 1
}

info() {
    echo "${cyan}$1${reset}"
}

warn() {
    echo "${yellow}$1${reset}"
}

success() {
    echo "${green}$1${reset}"
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
    return $?
}

need_cmd() {
    if ! check_cmd "$1"
    then err "need '$1' (command not found)"
    fi
}

need_python() {
    # check if either python3 or python is available
    if ! check_cmd python && ! check_cmd python3
    then err "need 'python' or 'python3' (command not found)"
    fi
}

downloader() {
    if check_cmd curl
    then curl -sSfL "$1"
    elif check_cmd wget
    then wget -qO- "$1"
    else need_cmd "curl or wget"
    fi
}

get_python_command() {
    need_python

    # check if either python3 or python is available
    # and echo the python one that works
    if check_cmd python3
    then echo "python3"
    elif check_cmd python
    then echo "python"
    fi
}

check_home_path() {
    # check if a path exists as ~/path or $HOME/path
    if echo $PATH | grep -q "$HOME/$1" || echo $PATH | grep -q "~/$1"
    then return 0
    else return 1
    fi
}

write_path() {
    local _path_contents="$1"
    local _profile_path="$2"
    # if profile exists, add the export
    if [ -f "$_profile_path" ]
    then
        echo "export PATH=\"$_path_contents\$PATH\"" >> $_profile_path;
    fi
}

patch_path() {
    local _path_expr=""

    if ! check_home_path ".cargo/bin"
    then _path_expr="$HOME/.cargo/bin:"
    fi

    if ! check_home_path ".local/bin"
    then _path_expr="${_path_expr}$HOME/.local/bin:"
    fi

    # reload env vars
    export PATH="$_path_expr$PATH"

    # write to profile files
    write_path $_path_expr "$HOME/.profile"
    write_path $_path_expr "$HOME/.zshrc"
    write_path $_path_expr "$HOME/.bashrc"
    write_path $_path_expr "$HOME/.bash_profile"
}

download_uv() {
    if ! check_cmd "uv"
    then downloader https://astral.sh/uv/install.sh | env INSTALLER_PRINT_QUIET=1 sh
    fi
}

install_uv() {
    download_uv
    patch_path
}

install_syftbox() {
    need_cmd "uv"

    python_flag=""
    py_detail=""

    if [ $MANAGED_PYTHON -eq 1 ]
    then
        python_flag="--python $MANAGED_PYTHON_VERSION"
        py_detail="managed Python $MANAGED_PYTHON_VERSION"
    else
        py=$(get_python_command)
        python_flag="--python-preference system"
        py_detail="system $($py -V)"
    fi

    info "Installing SyftBox (with $py_detail)"

    exit=$(uv tool install $python_flag -Uq --force syftbox)
    if ! $(exit)
    then err "Failed to install SyftBox"
    fi
}

check_python_version() {
    # Try python3, if it exists; otherwise, fall back to python
    py=$(get_python_command)

    # Check if py is empty (meaning no python was found)
    if [ -z "$py" ]; then
        return 1
    fi

    # Check if the python version is >= 3.9
    py_valid_ver=$($py -c "import sys; print((sys.version_info[:2] >= ($REQ_PYTHON_MAJOR, $REQ_PYTHON_MINOR)))")

    # Check if Python version not is greater than or equal to 3.9
    if [ "$py_valid_ver" = "False" ]; then
        err "SyftBox requires Python $REQ_PYTHON_MAJOR.$REQ_PYTHON_MINOR or higher, found $($py -V). Please upgrade your Python installation and retry."
    fi
}

show_debug_and_exit() {
    need_python
    py=$(get_python_command)
    echo
    warn "SYFTBOX INSTALLER DEBUG REPORT"
    echo
    info System
    echo "SHELL           : $SHELL"
    echo "LANG            : $LANG"
    echo "LD_LIBRARY_PATH : $LD_LIBRARY_PATH"
    echo "PATH            : $PATH"
    echo "PWD             : $(pwd)"
    echo
    info "Python"
    echo "Alias           : $py"
    echo "Version         : $($py -V)"
    echo "which           : $(which $py)"
    echo "python env vars"
    env | grep -E "(PYTHON|PIP)" || echo "-"
    echo
    info "Python Virtual Environment"
    env | grep -E "(VIRTUAL_ENV|VENV)" || echo "-"
    echo
    info "Conda Environment"
    env | grep CONDA || echo "-"
    echo
    info "Python Runtime"
    $py -c '
import sys; \
print("sys.version      :", sys.version); \
print("sys.executable   :", sys.executable); \
print("sys.prefix       :", sys.prefix); \
print("sys.base_prefix  :", sys.base_prefix); \
print("sys.path"); \
[print("-", p) for p in sys.path];'
    exit 0
}

pre_install() {

    # check if python version is >= 3.9, if uv is not managing python
    if [ $MANAGED_PYTHON -eq 0 ]
    then check_python_version
    fi

    # if you see this message, you're good to go
    echo "
 ____         __ _   ____
/ ___| _   _ / _| |_| __ )  _____  __
\___ \| | | | |_| __|  _ \ / _ \ \/ /
 ___) | |_| |  _| |_| |_) | (_) >  <
|____/ \__, |_|  \__|____/ \___/_/\_\\
       |___/
"
}

run_client() {
    echo
    success "Starting SyftBox client..."
    exec ~/.local/bin/syftbox client < /dev/tty
}

prompt_run_client() {
    # prompt if they want to start the client
    echo
    prompt=$(echo "${yellow}Start the client now? [y/n]  ${reset}")
    while [ "$start_client" != "y" ] && [ "$start_client" != "Y" ] && [ "$start_client" != "n" ] && [ "$start_client" != "N" ]
    do
        read -p "$prompt" start_client < /dev/tty
    done

    if [ "$start_client" = "y" ] || [ "$start_client" = "Y" ]
    then run_client
    else prompt_restart_shell
    fi
}

prompt_restart_shell() {
    echo
    warn "RESTART your shell or RELOAD shell profile"
    echo "  \`source ~/.zshrc\`        (for zsh)"
    echo "  \`source ~/.bash_profile\` (for bash)"
    echo "  \`source ~/.profile\`      (for sh)"

    success "\nAfter reloading, start the client"
    echo "  \`syftbox client\`"
}

post_install() {
    if [ $RUN_CLIENT -eq 1 ]
    then run_client
    elif [ $ASK_RUN_CLIENT -eq 1 ]
    then prompt_run_client
    else prompt_restart_shell
    fi
}

do_install() {
    for arg in "$@"; do
        case "$arg" in
            -r|--run|run)
                RUN_CLIENT=1
                ;;
            -n|--no-prompt|no-prompt)
                ASK_RUN_CLIENT=0
                ;;
            --system-python|system-python)
                MANAGED_PYTHON=0
                ;;
            --managed-python|managed-python)
                MANAGED_PYTHON=1
                ;;
            --debug|debug)
                show_debug_and_exit
                ;;
            *)
                ;;
        esac
    done

    pre_install

    info "Installing uv"
    install_uv

    install_syftbox

    success "Installation completed!"
    post_install
}

do_install "$@" || exit 1
