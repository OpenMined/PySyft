from ethereum import utils
from ethereum import transactions
import os
import rlp
import requests

host = "http://127.0.0.1:3000"


def add_experiment(experimentAddress, jobAddresses, priv_key=None,
                   account_address=None, returnAbi=False):
    payload = {'experimentAddress': experimentAddress,
               'jobAddresses': jobAddresses, 'returnAbi': returnAbi,
               'accountAddress': account_address}

    r = requests.post('{}/experiment'.format(host), json=payload)
    print("/experiment", r)

    if returnAbi:
        json = r.json()
        return send_raw_transaction(json, priv_key)

    return r.status_code


def get_available_job_id():
    r = requests.get(host + "/availableJobId")

    print("/availableJobId", r)

    if 'jobId' not in r.json():
        return None

    job_id = r.json()['jobId']
    if job_id == '':
        return None

    return job_id


def get_job():
    job_id = get_available_job_id()
    if job_id is None:
        return None

    r = requests.get('{}/job/{}'.format(host, job_id))

    print("/job/" + job_id, r)

    return r.json()['jobAddress']


def add_result(jobAddress, resultAddress, priv_key=None,
               account_address=None, returnAbi=False):
    payload = {'jobAddress': jobAddress, 'resultAddress': resultAddress,
               'returnAbi': returnAbi, 'accountAddress': account_address}

    r = requests.post(host + "/result", json=payload)
    print("/result", r)

    if returnAbi:
        json = r.json()
        return send_raw_transaction(json, priv_key)

    return r.status_code


def get_result(jobAddress):
    r = requests.get(host + "/results/" + jobAddress)

    print("/results/" + jobAddress, r)
    addr = r.json()['resultAddress']
    if addr == "":
        return None

    return addr


def send_raw_transaction(json, priv_key):
    abi = utils.decode_hex(json['abi'][2:])
    nonce = json['nonce']
    gas = json['estimatedGas']
    contractAddress = json['contractAddress']

    transaction = sign_transaction(nonce, abi, priv_key, gas, contractAddress)
    transaction = '0x' + transaction
    payload = {'rawTransaction': transaction}
    r = requests.post(host + "/raw", json=payload)
    print("/raw", r)

    return r.status_code


def create_wallet():
    # TODO is urandom secure??? maybe need to use something else
    private_key = utils.sha3(os.urandom(4096))

    # TODO encrypt and store these keys somewhere for user?!?!
    raw_address = utils.privtoaddr(private_key)
    account_address = utils.checksum_encode(raw_address)

    print("DO NOT LOSS PRIVATE KEY OR PUBLIC KEY NO WAY TO RECOVER")
    print("PRIVATE KEY", private_key.hex(), "PUBLIC KEY", account_address)

    return private_key, account_address


def sign_transaction(nonce, abi, priv_key, gas, to):
    # TODO send gas price from bygone?!
    gasprice = 18000000000

    # [nonce, gasprice, startgas, to, value, data, v, r, s]
    tx = transactions.Transaction(nonce, gasprice, gas, to, 5, abi)
    signed_tx = tx.sign(priv_key)

    ret = rlp.encode(signed_tx).hex()
    return ret
