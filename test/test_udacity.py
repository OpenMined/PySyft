import torch


def test_section_1_differential_privacy():
    """This tests the Udacity course content found at
    https://github.com/Udacity/private-ai
    """

    # the number of entries in our database
    num_entries = 5000

    db = torch.rand(num_entries) > 0.5

    db = torch.rand(num_entries) > 0.5

    def get_parallel_db(db, remove_index):
        return torch.cat((db[0:remove_index], db[remove_index + 1 :]))

    get_parallel_db(db, 52352)

    def get_parallel_dbs(db):
        parallel_dbs = []

        for i in range(len(db)):
            pdb = get_parallel_db(db, i)
            parallel_dbs.append(pdb)

        return parallel_dbs

    pdbs = get_parallel_dbs(db)

    def create_db_and_parallels(num_entries):
        db = torch.rand(num_entries) > 0.5
        pdbs = get_parallel_dbs(db)

        return db, pdbs

    db, pdbs = create_db_and_parallels(20)

    db, pdbs = create_db_and_parallels(5000)

    def query(db):
        return db.sum()

    full_db_result = query(db)

    sensitivity = 0
    for pdb in pdbs:
        pdb_result = query(pdb)

        db_distance = torch.abs(pdb_result - full_db_result)

        if db_distance > sensitivity:
            sensitivity = db_distance

    def sensitivity(query, n_entries=1000):

        db, pdbs = create_db_and_parallels(n_entries)

        full_db_result = query(db)

        max_distance = 0
        for pdb in pdbs:
            pdb_result = query(pdb)

            db_distance = torch.abs(pdb_result - full_db_result)

            if db_distance > max_distance:
                max_distance = db_distance

        return max_distance

    def query(db):
        return db.float().mean()

    sensitivity(query)

    db, pdbs = create_db_and_parallels(20)

    db

    def query(db, threshold=5):
        return (db.sum() > threshold).float()

    for i in range(10):
        sens_f = sensitivity(query, n_entries=10)
        print(sens_f)

    db, _ = create_db_and_parallels(100)

    pdb = get_parallel_db(db, remove_index=10)

    db[10]

    sum(db)

    # differencing attack using sum query

    sum(db) - sum(pdb)

    # differencing attack using mean query

    (sum(db).float() / len(db)) - (sum(pdb).float() / len(pdb))

    # differencing attack using threshold

    (sum(db).float() > 49).float() - (sum(pdb).float() > 49).float()

    def query(db):

        true_result = torch.mean(db.float())

        first_coin_flip = (torch.rand(len(db)) > 0.5).float()
        second_coin_flip = (torch.rand(len(db)) > 0.5).float()

        augmented_database = db.float() * first_coin_flip + (1 - first_coin_flip) * second_coin_flip

        db_result = torch.mean(augmented_database.float()) * 2 - 0.5

        return db_result, true_result

    db, pdbs = create_db_and_parallels(10)
    private_result, true_result = query(db)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(100)
    private_result, true_result = query(db)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(1000)
    private_result, true_result = query(db)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(10000)
    private_result, true_result = query(db)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    def query(db, noise=0.2):

        true_result = torch.mean(db.float())

        first_coin_flip = (torch.rand(len(db)) > noise).float()
        second_coin_flip = (torch.rand(len(db)) > 0.5).float()

        augmented_database = db.float() * first_coin_flip + (1 - first_coin_flip) * second_coin_flip

        sk_result = augmented_database.float().mean()

        private_result = ((sk_result / noise) - 0.5) * noise / (1 - noise)

        return private_result, true_result

    db, pdbs = create_db_and_parallels(100)
    private_result, true_result = query(db, noise=0.1)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(100)
    private_result, true_result = query(db, noise=0.2)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(100)
    private_result, true_result = query(db, noise=0.4)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(100)
    private_result, true_result = query(db, noise=0.8)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(10000)
    private_result, true_result = query(db, noise=0.8)
    print("With Noise:" + str(private_result))
    print("Without Noise:" + str(true_result))

    db, pdbs = create_db_and_parallels(100)

    def query(db):
        return torch.sum(db.float())

    # def M(db):
    #     query(db)  # + noise
    #
    query(db)

    epsilon = 0.0001

    import numpy as np

    db, pdbs = create_db_and_parallels(100)

    def sum_query(db):
        return db.sum()

    def laplacian_mechanism(db, query, sensitivity):

        beta = sensitivity / epsilon
        noise = torch.tensor(np.random.laplace(0, beta, 1))

        return query(db) + noise

    def mean_query(db):
        return torch.mean(db.float())

    laplacian_mechanism(db, sum_query, 1)

    laplacian_mechanism(db, mean_query, 1 / 100)

    import numpy as np

    num_teachers = 10  # we're working with 10 partner hospitals
    num_examples = 10000  # the size of OUR dataset
    num_labels = 10  # number of lablels for our classifier

    preds = (
        (np.random.rand(num_teachers, num_examples) * num_labels).astype(int).transpose(1, 0)
    )  # fake predictions

    new_labels = []
    for an_image in preds:

        label_counts = np.bincount(an_image, minlength=num_labels)

        epsilon = 0.1
        beta = 1 / epsilon

        for i in range(len(label_counts)):
            label_counts[i] += np.random.laplace(0, beta, 1)

        new_label = np.argmax(label_counts)

        new_labels.append(new_label)

    labels = np.array([9, 9, 3, 6, 9, 9, 9, 9, 8, 2])
    counts = np.bincount(labels, minlength=10)
    query_result = np.argmax(counts)
    query_result

    from syft.frameworks.torch.dp import pate

    num_teachers, num_examples, num_labels = (100, 100, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds
    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis(
        teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5
    )

    assert data_dep_eps < data_ind_eps

    data_dep_eps, data_ind_eps = pate.perform_analysis(
        teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5
    )
    print("Data Independent Epsilon:", data_ind_eps)
    print("Data Dependent Epsilon:", data_dep_eps)

    preds[:, 0:50] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis(
        teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5, moments=20
    )
    print("Data Independent Epsilon:", data_ind_eps)
    print("Data Dependent Epsilon:", data_dep_eps)


def test_section_2_federated_learning(hook):
    """This tests the Udacity course content found at
    https://github.com/Udacity/private-ai
    """

    import torch as th

    x = th.tensor([1, 2, 3, 4, 5])
    x

    y = x + x

    print(y)

    import syft as sy

    # commented out because the test needs to use the global one
    # hook = sy.TorchHook(th)

    th.tensor([1, 2, 3, 4, 5])

    bob = sy.VirtualWorker(hook, id="bob_udacity")

    bob._tensors

    x = th.tensor([1, 2, 3, 4, 5])

    x = x.send(bob)

    bob._tensors

    assert len(bob._tensors) == 1

    x.location

    x.id_at_location

    x.id

    x.owner

    hook.local_worker

    x

    x = x.get()
    x

    bob._tensors

    assert len(bob._tensors) == 0

    alice = sy.VirtualWorker(hook, id="alice_udacity")

    x = th.tensor([1, 2, 3, 4, 5])

    x_ptr = x.send(bob, alice)

    x_ptr.get()

    x = th.tensor([1, 2, 3, 4, 5]).send(bob, alice)

    x.get(sum_results=True)

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    y = th.tensor([1, 1, 1, 1, 1]).send(bob)

    x
    y

    z = x + y

    z

    z = z.get()
    z

    z = th.add(x, y)
    z

    z = z.get()
    z

    x = th.tensor([1.0, 2, 3, 4, 5], requires_grad=True).send(bob)
    y = th.tensor([1.0, 1, 1, 1, 1], requires_grad=True).send(bob)

    z = (x + y).sum()

    z.backward()

    x = x.get()

    x

    x.grad

    input = th.tensor([[1.0, 1], [0, 1], [1, 0], [0, 0]], requires_grad=True).send(bob)
    target = th.tensor([[1.0], [1], [0], [0]], requires_grad=True).send(bob)

    weights = th.tensor([[0.0], [0.0]], requires_grad=True).send(bob)

    for i in range(10):
        pred = input.mm(weights)

        loss = ((pred - target) ** 2).sum()

        loss.backward()

        weights.data.sub_(weights.grad * 0.1)
        weights.grad *= 0

        print(loss.get().data)

    bob = bob.clear_objects()

    assert len(bob._objects) == 0

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)

    assert len(bob._objects) == 1

    del x

    assert len(bob._objects) == 0

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)

    assert len(bob._objects) == 1

    x = "asdf"

    assert len(bob._objects) == 0

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)

    bob._objects

    x = "asdf"

    bob._objects

    del x

    bob._objects

    bob = bob.clear_objects()
    bob._objects

    for i in range(1000):
        x = th.tensor([1, 2, 3, 4, 5]).send(bob)

    assert len(bob._objects) == 1

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    y = th.tensor([1, 1, 1, 1, 1])

    # throws error
    # z = x + y

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)
    y = th.tensor([1, 1, 1, 1, 1]).send(alice)

    # throws error
    # z = x + y

    from torch import nn, optim

    # A Toy Dataset
    data = th.tensor([[1.0, 1], [0, 1], [1, 0], [0, 0]], requires_grad=True)
    target = th.tensor([[1.0], [1], [0], [0]], requires_grad=True)

    # A Toy Model
    model = nn.Linear(2, 1)

    opt = optim.SGD(params=model.parameters(), lr=0.1)

    def train(iterations=20):
        for iter in range(iterations):
            opt.zero_grad()

            pred = model(data)

            loss = ((pred - target) ** 2).sum()

            loss.backward()

            opt.step()

            print(loss.data)

    train()

    data_bob = data[0:2].send(bob)
    target_bob = target[0:2].send(bob)

    data_alice = data[2:4].send(alice)
    target_alice = target[2:4].send(alice)

    datasets = [(data_bob, target_bob), (data_alice, target_alice)]

    def train(iterations=20):

        model = nn.Linear(2, 1)
        opt = optim.SGD(params=model.parameters(), lr=0.1)

        for iter in range(iterations):

            for _data, _target in datasets:
                # send model to the data
                model = model.send(_data.location)

                # do normal training
                opt.zero_grad()
                pred = model(_data)
                loss = ((pred - _target) ** 2).sum()
                loss.backward()
                opt.step()

                # get smarter model back
                model = model.get()

                print(loss.get())

    train()

    bob.clear_objects()
    alice.clear_objects()

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)

    x = x.send(alice)

    bob._objects

    alice._objects

    y = x + x

    y

    bob._objects

    alice._objects

    jon = sy.VirtualWorker(hook, id="jon")

    bob.clear_objects()
    alice.clear_objects()

    x = th.tensor([1, 2, 3, 4, 5]).send(bob).send(alice)

    bob._objects

    alice._objects

    x = x.get()
    x

    bob._objects

    alice._objects

    x = x.get()
    x

    bob._objects

    bob.clear_objects()
    alice.clear_objects()

    x = th.tensor([1, 2, 3, 4, 5]).send(bob).send(alice)

    bob._objects

    alice._objects

    del x

    bob._objects

    alice._objects

    bob.clear_objects()
    alice.clear_objects()

    x = th.tensor([1, 2, 3, 4, 5]).send(bob)

    bob._objects

    alice._objects

    x.move(alice)

    bob._objects

    alice._objects

    x = th.tensor([1, 2, 3, 4, 5]).send(bob).send(alice)

    bob._objects

    alice._objects

    x.remote_get()

    bob._objects

    alice._objects

    x.move(bob)

    x

    bob._objects

    alice._objects


def test_section_3_securing_fl(hook):
    """This tests the Udacity course content found at
    https://github.com/Udacity/private-ai
    """

    import syft as sy
    import torch as th

    # hook = sy.TorchHook(th)
    from torch import nn, optim

    # create a couple workers

    bob = sy.VirtualWorker(hook, id="bob_udacity_3")
    alice = sy.VirtualWorker(hook, id="alice_udacity_3")
    secure_worker = sy.VirtualWorker(hook, id="secure_worker_udacity_3")

    bob.add_workers([alice, secure_worker])
    alice.add_workers([bob, secure_worker])
    secure_worker.add_workers([alice, bob])

    # A Toy Dataset
    data = th.tensor([[0, 0], [0, 1], [1, 0], [1, 1.0]], requires_grad=True)
    target = th.tensor([[0], [0], [1], [1.0]], requires_grad=True)

    # get pointers to training data on each worker by
    # sending some training data to bob and alice
    bobs_data = data[0:2].send(bob)
    bobs_target = target[0:2].send(bob)

    alices_data = data[2:].send(alice)
    alices_target = target[2:].send(alice)

    # Iniitalize A Toy Model
    model = nn.Linear(2, 1)

    bobs_model = model.copy().send(bob)
    alices_model = model.copy().send(alice)

    bobs_opt = optim.SGD(params=bobs_model.parameters(), lr=0.1)
    alices_opt = optim.SGD(params=alices_model.parameters(), lr=0.1)

    for i in range(10):
        # Train Bob's Model
        bobs_opt.zero_grad()
        bobs_pred = bobs_model(bobs_data)
        bobs_loss = ((bobs_pred - bobs_target) ** 2).sum()
        bobs_loss.backward()

        bobs_opt.step()
        bobs_loss = bobs_loss.get().data

        # Train Alice's Model
        alices_opt.zero_grad()
        alices_pred = alices_model(alices_data)
        alices_loss = ((alices_pred - alices_target) ** 2).sum()
        alices_loss.backward()

        alices_opt.step()
        alices_loss = alices_loss.get().data
        alices_loss

    alices_model.move(secure_worker)
    bobs_model.move(secure_worker)

    with th.no_grad():

        model.weight.set_(((alices_model.weight.data + bobs_model.weight.data) / 2).get())
        model.bias.set_(((alices_model.bias.data + bobs_model.bias.data) / 2).get())

    iterations = 10
    worker_iters = 5

    for a_iter in range(iterations):

        bobs_model = model.copy().send(bob)
        alices_model = model.copy().send(alice)

        bobs_opt = optim.SGD(params=bobs_model.parameters(), lr=0.1)
        alices_opt = optim.SGD(params=alices_model.parameters(), lr=0.1)

        for wi in range(worker_iters):
            # Train Bob's Model
            bobs_opt.zero_grad()
            bobs_pred = bobs_model(bobs_data)
            bobs_loss = ((bobs_pred - bobs_target) ** 2).sum()
            bobs_loss.backward()

            bobs_opt.step()
            bobs_loss = bobs_loss.get().data

            # Train Alice's Model
            alices_opt.zero_grad()
            alices_pred = alices_model(alices_data)
            alices_loss = ((alices_pred - alices_target) ** 2).sum()
            alices_loss.backward()

            alices_opt.step()
            alices_loss = alices_loss.get().data

        alices_model.move(secure_worker)
        bobs_model.move(secure_worker)

        with th.no_grad():

            model.weight.set_(((alices_model.weight.data + bobs_model.weight.data) / 2).get())
            model.bias.set_(((alices_model.bias.data + bobs_model.bias.data) / 2).get())

        print("Bob:" + str(bobs_loss) + " Alice:" + str(alices_loss))

    preds = model(data)
    loss = ((preds - target) ** 2).sum()

    print(preds)
    print(target)
    print(loss.data)

    x = 5

    bob_x_share = 2
    alice_x_share = 3

    decrypted_x = bob_x_share + alice_x_share
    decrypted_x

    bob_x_share = 2 * 2
    alice_x_share = 3 * 2

    decrypted_x = bob_x_share + alice_x_share
    decrypted_x

    # encrypted "5"
    bob_x_share = 2
    alice_x_share = 3

    # encrypted "7"
    bob_y_share = 5
    alice_y_share = 2

    # encrypted 5 + 7
    bob_z_share = bob_x_share + bob_y_share
    alice_z_share = alice_x_share + alice_y_share

    decrypted_z = bob_z_share + alice_z_share
    decrypted_z

    x = 5

    Q = 23740629843760239486723

    bob_x_share = 23552870267  # <- a random number
    alice_x_share = Q - bob_x_share + x
    alice_x_share

    (bob_x_share + alice_x_share) % Q

    x_share = (2, 5, 7)

    import random

    Q = 23740629843760239486723

    def encrypt(x, n_share=3):

        shares = []

        for i in range(n_share - 1):
            shares.append(random.randint(0, Q))

        shares.append(Q - (sum(shares) % Q) + x)

        return tuple(shares)

    def decrypt(shares):
        return sum(shares) % Q

    shares = encrypt(3)
    shares

    decrypt(shares)

    def add(a, b):
        c = []
        for i in range(len(a)):
            c.append((a[i] + b[i]) % Q)
        return tuple(c)

    x = encrypt(5)
    y = encrypt(7)
    z = add(x, y)
    decrypt(z)

    BASE = 10
    PRECISION = 4

    def encode(x):
        return int((x * (BASE ** PRECISION)) % Q)

    def decode(x):
        return (x if x <= Q / 2 else x - Q) / BASE ** PRECISION

    encode(3.5)

    decode(35000)

    x = encrypt(encode(5.5))
    y = encrypt(encode(2.3))
    z = add(x, y)
    decode(decrypt(z))

    bob = bob.clear_objects()
    alice = alice.clear_objects()
    secure_worker = secure_worker.clear_objects()

    x = th.tensor([1, 2, 3, 4, 5])

    x = x.share(bob, alice, secure_worker)

    bob._objects

    y = x + x

    y

    y.get()

    x = th.tensor([0.1, 0.2, 0.3])

    x = x.fix_prec()

    x.child.child

    y = x + x

    y = y.float_prec()
    y

    x = th.tensor([0.1, 0.2, 0.3])

    x = x.fix_prec().share(bob, alice, secure_worker)

    y = x + x

    y.get().float_prec()
