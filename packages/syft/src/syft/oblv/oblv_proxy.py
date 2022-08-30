# stdlib
import os
import platform
from re import sub
import subprocess
import zipfile

# third party
import requests

# from ..logger import debug


def check_oblv_proxy_installation_status():
    try:
        result = subprocess.run(["oblv","-V"], capture_output=True, text=True)
        if result.stderr:
            raise subprocess.CalledProcessError(
                    returncode = result.returncode,
                    cmd = result.args,
                    stderr = result.stderr
                    )
        result = result.stdout.strip()
        return result
    except Exception as e:
        if e.__class__==FileNotFoundError:
            system_name = platform.system()
            result = "Oblv Proxy Not Installed. Call the method install_oblv_proxy "
            if system_name=="Windows":
                result += "to install the proxy for this session. If you already have the proxy installed, add it to your PATH." 
            elif system_name=="Linux":
                result += "to install the proxy globally. If you already have the proxy installed, create a link to the installation as /usr/local/bin/oblv" 
            return result
        raise Exception(e)   
    
def install_oblv_proxy():
    system_name = platform.system()
    if system_name=="Windows":
        windows_proxy_installation()
    elif system_name=="Linux":
        linux_proxy_installation()


def windows_proxy_installation():
    url='https://oblv-cli-binary.s3.us-east-2.amazonaws.com/0.3.0/oblv-ccli-0.3.0-x86_64-pc-windows-msvc.zip'
    res = requests.get(url)
    path = os.getcwd().replace('\\','/')+"/oblv-ccli-0.3.0-x86_64-pc-windows-msvc.zip"
    with open(path,"wb") as f:
        f.write(res.content)
    with  zipfile.ZipFile(path, 'r') as zipObj:
        zipObj.extractall()
    os.environ["PATH"] += os.getcwd() + "\\oblv-ccli-0.3.0-x86_64-pc-windows-msvc;"

def linux_proxy_installation():
    url='https://oblv-cli-binary.s3.us-east-2.amazonaws.com/0.3.0/oblv-ccli-0.3.0-x86_64-unknown-linux-musl.zip'
    res = requests.get(url)
    path = os.getcwd()+"/oblv-ccli-0.3.0-x86_64-unknown-linux-musl.zip"
    with open(path,"wb") as f:
        f.write(res.content)
    with  zipfile.ZipFile(path, 'r') as zipObj:
        zipObj.extractall()
    os.symlink('/usr/local/bin/oblv', os.getcwd()+"/oblv-ccli-0.3.0-x86_64-unknown-linux-musl/oblv")

# #ToDo
# def darwin_proxy_installation():
#     return
