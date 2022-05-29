# enter the dict given from the data owner
# third party
import numpy as np

# syft absolute
import syft as sy
from syft.core.tensor.smpc.mpc_tensor import MPCTensor


def add_credentials(credentials_dict_list):
    domain_credentials = set()
    for credentials_dict in credentials_dict_list:
        fs = frozenset(credentials_dict.items())
        domain_credentials.add(fs)

    return domain_credentials


def login_to_domains(domain_credentials: set, force: bool = False):
    domains = {}  # our logged in domain clients
    for domain_creds in domain_credentials:
        credentials = dict(domain_creds)
        if "host" in credentials:
            credentials["url"] = credentials["host"]
            del credentials["host"]
        if "budget" in credentials:
            del credentials["budget"]

        if credentials["url"] not in domains or force:
            try:
                details = credentials.copy()
                del details["name"]
                client = sy.login(**details)
                url = credentials["url"]
                domains[url] = client
            except Exception as e:
                print(e)

    return list(domains.values())


def preprocess_data_on_domains(domains):
    X_train, Y_train, X_val, Y_val = [], [], [], []
    for idx, domain in enumerate(domains):
        data = domain.datasets[-1]

        X_train.append(data["train_images"])
        Y_train.append(data["train_labels"])

        X_val.append(data["val_images"])
        Y_val.append(data["val_labels"])

        # pre process data
        X_train[idx] = (X_train[idx].T) * (1 / 255.0)
        X_val[idx] = (X_val[idx].T) * (1 / 255.0)

    return X_train, X_val, Y_train, Y_val


def init_params(input_size: int, label_size: int):
    print(f"Using input size: {input_size}")
    print(f"Using label size: {label_size}")
    W1 = np.random.rand(label_size, input_size) - 0.5
    b1 = np.random.rand(label_size, 1) - 0.5
    W2 = np.random.rand(label_size, label_size) - 0.5
    b2 = np.random.rand(label_size, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return Z * (Z > 0)


def softmax(Z):
    exp_cache = Z.exp()
    inv = exp_cache.sum().reciprocal()

    A = exp_cache * inv
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = X.__rmatmul__(W1) + b1
    A1 = ReLU(Z1)
    Z2 = A1.__rmatmul__(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    n, m = X.public_shape
    one_hot_Y = Y.one_hot()
    dZ2 = A2 - one_hot_Y
    dW2 = dZ2 @ (A1.T) * (1 / m)
    db2 = dZ2.sum() * (1 / m)
    dZ1 = dZ2.__rmatmul__(W2.T) * ReLU_deriv(Z1)
    dW1 = dZ1 @ (X.T) * (1 / m)
    db1 = dZ1.sum() * (1 / m)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = (dW1 * alpha - W1) * -1
    b1 = (db1 * alpha - b1) * -1
    W2 = (dW2 * alpha - W2) * -1
    b2 = (db2 * alpha - b2) * -1
    return W1, b1, W2, b2


def convert_to_mpc_tensor(*args):
    parties = [val.client for val in args[0]]
    for i in range(len(args)):
        for j in range(len(args[i])):
            args[i][j] = MPCTensor(
                secret=args[i][j], shape=args[i][j].public_shape, parties=parties
            )

    return args


def smpc_weight_averaging(W1, b1, W2, b2):
    W1, b1, W2, b2 = convert_to_mpc_tensor(W1, b1, W2, b2)
    print("finish")
    n = len(W1)
    avg_W1, avg_W2, avg_b1, avg_b2 = W1[0], W2[0], b1[0], b2[0]

    for i in range(1, n):
        avg_W1 = avg_W1 + W1[i]
        avg_W1 = avg_W2 + W2[i]
        avg_W1 = avg_b1 + b1[i]
        avg_W1 = avg_b2 + b2[i]
    print("Addition finish")

    avg_W1 = avg_W1 * (1 / n)
    print("First one finished")
    avg_W1 = avg_W2 * (1 / n)
    print("Second one finished")
    avg_b1 = avg_b1 * (1 / n)
    print("Third one finished")
    avg_b2 = avg_b2 * (1 / n)
    print("Fourth one finished")

    return avg_W1, avg_b1, avg_W2, avg_b2
