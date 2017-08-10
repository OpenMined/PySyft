def test_linear_classifier():

    from syft.he.Paillier import KeyPair
    from syft.nn.linear import LinearClassifier
    import numpy as np

    pubkey,prikey = KeyPair().generate(n_length=1024)

    model = LinearClassifier(n_inputs=4,n_labels=2).encrypt(pubkey)

    input = np.array([[0,0,1,1],[0,0,1,0],[1,0,1,1],[0,0,1,0]])
    target = np.array([[0,1],[0,0],[1,1],[0,0]])

    for iter in range(4):
        for i in range(len(input)):
            print("Grads:" + str((model.learn(input=input[i],target=target[i],alpha=0.5))))

    model = model.decrypt(prikey)
    for i in range(len(input)):
        print(model.forward(input[i]))
