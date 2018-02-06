from colorama import Fore, Back, Style
from grid import ipfsapi
import keras

def get_ipfs_api(ipfs_addr='127.0.0.1', port=5001):
    try:
        return ipfsapi.connect(ipfs_addr, port)
    except:
        print(f'\n{Fore.RED}ERROR: {Style.RESET_ALL}could not connect to IPFS.  Is your daemon running with pubsub support at {ipfs_addr} on port {port}')
        sys.exit()

def keras2ipfs(model):
    return get_ipfs_api().add_bytes(serialize_keras_model(model))

def ipfs2keras(model_addr):
    model_bin = get_ipfs_api().cat(model_addr)
    return deserialize_keras_model(model_bin)

def serialize_keras_model(model):
    model.save('temp_model.h5')
    with open('temp_model.h5','rb') as f:
        model_bin = f.read()
        f.close()
    return model_bin

def deserialize_keras_model(model_bin):
    with open('temp_model2.h5','wb') as g:
        g.write(model_bin)
        g.close()
    model = keras.models.load_model('temp_model2.h5')
    return model
