# stdlib
import base64
import os
import platform
from re import sub
import signal
import subprocess
import zipfile

# third party
from oblv import OblvClient
import requests

# relative
from ..core.node.common.exceptions import OblvEnclaveError
from ..core.node.common.exceptions import OblvEnclaveUnAuthorizedError
from ..core.node.common.exceptions import OblvError

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
            raise OblvError(result)
        raise Exception(e)   
    
def install_oblv_proxy():
    system_name = platform.system()
    if system_name=="Windows":
        windows_proxy_installation()
    elif system_name=="Linux":
        linux_proxy_installation()
    elif system_name=="Darwin":
        darwin_proxy_installation()

def windows_proxy_installation():
    try:
        url='https://oblv-cli-binary.s3.us-east-2.amazonaws.com/0.3.0/oblv-ccli-0.3.0-x86_64-pc-windows-msvc.zip'
        res = requests.get(url)
        path = os.getcwd().replace('\\','/')+"/oblv-ccli-0.3.0-x86_64-pc-windows-msvc.zip"
        with open(path,"wb") as f:
            f.write(res.content)
        with  zipfile.ZipFile(path, 'r') as zipObj:
            zipObj.extractall()
        os.environ["PATH"] += ";"+os.getcwd() + "\\oblv-ccli-0.3.0-x86_64-pc-windows-msvc;"
    except Exception as e:
        print(e)

def linux_proxy_installation():
    url='https://oblv-cli-binary.s3.us-east-2.amazonaws.com/0.3.0/oblv-ccli-0.3.0-x86_64-unknown-linux-musl.zip'
    res = requests.get(url)
    path = os.getcwd()+"/oblv-ccli-0.3.0-x86_64-unknown-linux-musl.zip"
    with open(path,"wb") as f:
        f.write(res.content)
    with  zipfile.ZipFile(path, 'r') as zipObj:
        zipObj.extractall()
    os.symlink('/usr/local/bin/oblv', os.getcwd()+"/oblv-ccli-0.3.0-x86_64-unknown-linux-musl/oblv")

def darwin_proxy_installation():
    url='https://oblv-cli-binary.s3.us-east-2.amazonaws.com/0.3.0/oblv-ccli-0.3.0-x86_64-apple-darwin.zip'
    res = requests.get(url)
    path = os.getcwd()+"/oblv-ccli-0.3.0-x86_64-apple-darwin.zip"
    with open(path,"wb") as f:
        f.write(res.content)
    with  zipfile.ZipFile(path, 'r') as zipObj:
        zipObj.extractall()
    ###Need to test this out
    os.symlink('/usr/local/bin/oblv', os.getcwd()+"/oblv-ccli-0.3.0-x86_64-apple-darwin/oblv")
    
def create_oblv_key_pair(key_name):
    check_oblv_proxy_installation_status()
    try:
        file_path=os.path.join(os.path.expanduser('~'),'.ssh',key_name)
        result = subprocess.run(["oblv", "keygen", "--key-name", key_name,"--output",file_path],capture_output=True)
        if result.stderr:
            raise subprocess.CalledProcessError(
                    returncode = result.returncode,
                    cmd = result.args,
                    stderr = result.stderr
                    )
        result = result.stdout.strip()
        print(result)
        return get_oblv_public_key(key_name)
    except Exception as e:
        raise Exception(e)   

def get_oblv_public_key(key_name):
    try:
        filepath = os.path.join(os.path.expanduser('~'),'.ssh',key_name,key_name+'_public.der')
        with open(filepath,'rb') as f:
            public_key=f.read()
        public_key = base64.encodebytes(public_key).decode("UTF-8").replace("\n","")
        return public_key
    except FileNotFoundError:
        print("No key found with given name")
        raise FileNotFoundError
    except Exception as e:
        raise Exception(e)
