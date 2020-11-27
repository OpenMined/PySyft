import os

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

HOME = "/Users/tryffel/code"


def get_number_classes(dataset):
    number_classes = {"mnist": 10, "cifar10": 10, "tiny-imagenet": 200, "hymenoptera": 2}
    return number_classes[dataset]


def one_hot_of(index_tensor):
    """
    Transform to one hot tensor

    Example:
        [0, 3, 9]
        =>
        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

    """
    onehot_tensor = torch.zeros(*index_tensor.shape, 10)  # 10 classes for MNIST
    onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
    return onehot_tensor


def get_data_loaders(args, kwargs, private=True):
    def encode(tensor):
        """
        Depending on the setting, acts on a tensor
        - Do nothing
        OR
        - Transform to fixed precision
        OR
        - Secret share
        """
        if args.public:
            return tensor

        encrypted_tensor = tensor.encrypt(**kwargs)
        if args.fp_only:  # Just keep the (Autograd+) Fixed Precision feature
            return encrypted_tensor.get()
        return encrypted_tensor

    dataset = args.dataset

    if dataset == "mnist":
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(
            "../data", train=True, download=True, transform=transformation
        )
        test_dataset = datasets.MNIST(
            "../data", train=False, download=True, transform=transformation
        )

    elif dataset == "cifar10":
        transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        train_dataset = datasets.CIFAR10(
            "../data", train=True, download=True, transform=transformation
        )
        test_dataset = datasets.CIFAR10(
            "../data", train=False, download=True, transform=transformation
        )

    elif dataset == "tiny-imagenet":
        transformation = transforms.Compose([transforms.ToTensor()])
        try:
            data_dir = HOME + "/pytorch-tiny-imagenet/tiny-imagenet-200/"
            train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transformation)
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transformation)
        except FileNotFoundError:
            raise FileNotFoundError(
                "You need to install manually the Tiny Imagenet dataset from GitHub.\n"
                "Instruction can be found here:\n"
                "https://github.com/tjmoon0104/pytorch-tiny-imagenet"
            )

    elif dataset == "hymenoptera":
        train_transformation = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        test_transformation = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        try:
            data_dir = HOME + "/hymenoptera_data"
            train_dataset = datasets.ImageFolder(
                os.path.join(data_dir, "train"), train_transformation
            )
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), test_transformation)
        except FileNotFoundError:
            raise FileNotFoundError(
                "You need to install manually the hymenoptera dataset.\n"
                f"Here are the instruction to run in {HOME}:\n"
                "wget https://download.pytorch.org/tutorial/hymenoptera_data.zip\n"
                "unzip hymenoptera_data.zip"
            )
    else:
        raise ValueError(f"Not supported dataset {dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
    )

    new_train_loader = []
    for i, (data, target) in enumerate(train_loader):
        if args.n_train_items >= 0 and i >= args.n_train_items / args.batch_size:
            break
        if private:
            new_train_loader.append((encode(data), encode(one_hot_of(target))))
        else:
            new_train_loader.append((data, one_hot_of(target)))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        drop_last=True,
    )

    new_test_loader = []
    for i, (data, target) in enumerate(test_loader):
        if args.n_test_items >= 0 and i >= args.n_test_items / args.test_batch_size:
            break
        if private:
            new_test_loader.append((encode(data), encode(target.float())))
        else:
            new_test_loader.append((data, target))

    return new_train_loader, new_test_loader
