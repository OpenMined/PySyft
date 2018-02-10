from ethereum import utils
from ethereum import tools
from ethereum import transactions
import rlp
import requests
from mnemonic import Mnemonic
import json
import getpass

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


def send_raw_transaction(json):
    print("ENTER KEYSTORE JSON FILE (default wallet.json):")
    keystore_file = input()
    if keystore_file == '':
        keystore_file = 'wallet.json'

    print("NEED PASSWORD TO UNLOCK WALLET:")
    password = getpass.getpass()
    priv_key = get_private_json_wallet(password, keystore_file=keystore_file)
    print(priv_key)

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


def create_wallet(keystore_pass, mnemonic_pass, mnemonic='',
                  keystore_file='wallet.json'):
    # TODO is this secure, maybe not trust this yet on a real network
    # SHOULD DEFINITELY NOT BE USED IN PRODUCTION YET
    m = Mnemonic('english')

    print(keystore_pass, mnemonic_pass)

    if mnemonic is '':
        # strength of 256 is recommended
        mnemonic = m.generate(strength=256)

    seed = m.to_seed(mnemonic, mnemonic_pass)

    private_key = utils.sha3(seed)

    raw_address = utils.privtoaddr(private_key)
    account_address = utils.checksum_encode(raw_address)

    print("WARNING: BELOW SHOULDN'T BE USED IN PRODUCTION, NOT AUDITED FOR SAFETY")
    print("SAVE MNEMONIC & PASSPHRASE OFFLINE TO RECOVER ACCOUNT")
    print("MNEMONIC:", mnemonic)
    print("PASSPHRASE:", mnemonic_pass)

    print("PRIVATE KEY", private_key.hex(), "PUBLIC KEY", account_address)

    print("PRIVATE KEY STORED IN", keystore_file, "WITH PASSWORD:", keystore_pass)
    print("CAN BE RECOVERED WITH ABOVE MNEMONIC AND PASSPHRASE")
    store_json_wallet(private_key, account_address, keystore_pass,
                      keystore_file)

    return private_key.hex(), account_address


def sign_transaction(nonce, abi, priv_key, gas, to):
    # TODO send gas price from bygone?!
    gasprice = 18000000000

    # [nonce, gasprice, startgas, to, value, data, v, r, s]
    tx = transactions.Transaction(nonce, gasprice, gas, to, 5, abi)
    signed_tx = tx.sign(priv_key)

    ret = rlp.encode(signed_tx).hex()
    return ret


def store_json_wallet(private_key, account, password, keystore_file):
    keystore = tools.keys.make_keystore_json(private_key, password)

    account = account.replace('0x', '')
    keystore['account'] = account
    keystore['id'] = keystore['id'].decode('utf-8')
    with open(keystore_file, 'w') as outfile:
        json.dump(keystore, outfile)


def get_private_json_wallet(password, keystore_file='wallet.json'):
    with open(keystore_file, 'r', ) as outfile:
        keystore = json.load(outfile)
        priv_key = tools.keys.decode_keystore_json(keystore, password)

    return priv_key
