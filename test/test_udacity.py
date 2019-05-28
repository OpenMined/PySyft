import pytest
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
        return torch.cat((db[0:remove_index],
                          db[remove_index + 1:]))

    get_parallel_db(db, 52352)

    def get_parallel_dbs(db):
        parallel_dbs = list()

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

        if (db_distance > sensitivity):
            sensitivity = db_distance

    def sensitivity(query, n_entries=1000):

        db, pdbs = create_db_and_parallels(n_entries)

        full_db_result = query(db)

        max_distance = 0
        for pdb in pdbs:
            pdb_result = query(pdb)

            db_distance = torch.abs(pdb_result - full_db_result)

            if (db_distance > max_distance):
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

    (sum(db).float() > 49) - (sum(pdb).float() > 49)

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

    def M(db):
        query(db) + noise

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

    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int).transpose(1, 0)  # fake predictions

    new_labels = list()
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

    from syft.frameworks.torch.differential_privacy import pate

    num_teachers, num_examples, num_labels = (100, 100, 10)
    preds = (np.random.rand(num_teachers, num_examples) * num_labels).astype(int)  # fake preds
    indices = (np.random.rand(num_examples) * num_labels).astype(int)  # true answers

    preds[:, 0:10] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5)

    assert data_dep_eps < data_ind_eps

    data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5)
    print("Data Independent Epsilon:", data_ind_eps)
    print("Data Dependent Epsilon:", data_dep_eps)

    preds[:, 0:50] *= 0

    data_dep_eps, data_ind_eps = pate.perform_analysis(teacher_preds=preds, indices=indices, noise_eps=0.1, delta=1e-5,
                                                       moments=20)
    print("Data Independent Epsilon:", data_ind_eps)
    print("Data Dependent Epsilon:", data_dep_eps)

    assert True
