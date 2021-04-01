# stdlib
from typing import Dict
from typing import Union

# TODO: Refactor out all the test data.
# Issue: https://github.com/OpenMined/PySyft/issues/5325

allowlist: Dict[str, Union[str, Dict[str, str]]] = {}  # (path: str, return_type:type)

allowlist["torchvision.__version__"] = "syft.lib.python.String"
# MNIST
allowlist["torchvision.transforms.Compose"] = "torchvision.transforms.Compose"
# allowlist["torchvision.transforms.Compose.__iter__"] = "torchvision.transforms.ToTensor"
# TODO: Compose.transforms property only exists on the object not on the class?
# allowlist["torchvision.transforms.Compose.transforms"] = "syft.lib.python.List"
allowlist["torchvision.transforms.ToTensor"] = "torchvision.transforms.ToTensor"
allowlist["torchvision.transforms.Normalize"] = "torchvision.transforms.Normalize"
# TODO: Normalize properties only exists on the object not on the class?
# allowlist["torchvision.transforms.Normalize.inplace"] = "syft.lib.python.Bool"
# TODO: mean and std are actually tuples
# allowlist["torchvision.transforms.Normalize.mean"] = "syft.lib.python.List"
# allowlist["torchvision.transforms.Normalize.std"] = "syft.lib.python.List"

# Datasets

allowlist["torchvision.datasets.MNIST"] = {
    "return_type": "torchvision.datasets.MNIST",
    "test_parameters": "('../data', download=False,)",
}


allowlist["torchvision.datasets.MNIST.__len__"] = "syft.lib.python.Int"


allowlist["torchvision.datasets.CelebA"] = {
    "return_type": "torchvision.datasets.CelebA",
    "test_parameters": "('../data')",
}
allowlist["torchvision.datasets.CelebA.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CIFAR10"] = {
    "return_type": "torchvision.datasets.CIFAR10",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.CIFAR10.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CIFAR100"] = {
    "return_type": "torchvision.datasets.CIFAR100",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.CIFAR10.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Cityscapes"] = {
    "return_type": "torchvision.datasets.Cityscapes",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.Cityscapes.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CocoCaptions"] = {
    "return_type": "torchvision.datasets.CocoCaptions",
    "test_parameters": "('../data','../data/captions.txt')",
}
allowlist["torchvision.datasets.CocoCaptions.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CocoDetection"] = {
    "return_type": "torchvision.datasets.CocoDetection",
    "test_parameters": "('../data', '../data/captions.txt')",
}
allowlist["torchvision.datasets.CocoDetection.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.EMNIST"] = {
    "return_type": "torchvision.datasets.EMNIST",
    "test_parameters": "('../data',split = \"mnist\")",
}
allowlist["torchvision.datasets.EMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.FakeData"] = {
    "return_type": "torchvision.datasets.FakeData",
    "test_parameters": "('../data', )",
}
allowlist["torchvision.datasets.FakeData.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.FashionMNIST"] = {
    "return_type": "torchvision.datasets.FashionMNIST",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.FashionMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Flickr8k"] = {
    "return_type": "torchvision.datasets.Flickr8k",
    "test_parameters": "('../data', '../data/annfile.txt')",
}
allowlist["torchvision.datasets.Flickr8k.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Flickr30k"] = {
    "return_type": "torchvision.datasets.Flickr30k",
    "test_parameters": "('../data', '../data/annfile.txt')",
}
allowlist["torchvision.datasets.Flickr30k.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.HMDB51"] = {
    "return_type": "torchvision.datasets.HMDB51",
    "test_parameters": "('../data', '../data/annfile.txt', 20,  )",
}
allowlist["torchvision.datasets.HMDB51.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.ImageNet"] = {
    "return_type": "torchvision.datasets.ImageNet",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.ImageNet.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Kinetics400"] = {
    "return_type": "torchvision.datasets.Kinetics400",
    "test_parameters": "('../data', 20)",
}
allowlist["torchvision.datasets.Kinetics400.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.KMNIST"] = {
    "return_type": "torchvision.datasets.KMNIST",
    "test_parameters": "('../data', )",
}
allowlist["torchvision.datasets.KMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.LSUN"] = {
    "return_type": "torchvision.datasets.LSUN",
    "test_parameters": "('../data', )",
}
allowlist["torchvision.datasets.LSUN.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Omniglot"] = {
    "return_type": "torchvision.datasets.Omniglot",
    "min_version": "0.8.0",
    "test_parameters": "('../data', )",
}
allowlist["torchvision.datasets.Omniglot.__len__"] = {
    "return_type": "syft.lib.python.Int",
    "min_version": "0.8.0",
}

allowlist["torchvision.datasets.PhotoTour"] = {
    "return_type": "torchvision.datasets.PhotoTour",
    "test_parameters": "('../data', name = 'data')",
}
allowlist["torchvision.datasets.PhotoTour.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Places365"] = {
    "return_type": "torchvision.datasets.Places365",
    "min_version": "0.8.0",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.Places365.__len__"] = {
    "return_type": "syft.lib.python.Int",
    "min_version": "0.8.0",
}

allowlist["torchvision.datasets.QMNIST"] = {
    "return_type": "torchvision.datasets.QMNIST",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.QMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.SBDataset"] = {
    "return_type": "torchvision.datasets.SBDataset",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.SBDataset.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.SBU"] = {
    "return_type": "torchvision.datasets.SBU",
    "test_parameters": "('../data', download = False)",
}
allowlist["torchvision.datasets.SBU.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.STL10"] = {
    "return_type": "torchvision.datasets.STL10",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.STL10.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.SVHN"] = {
    "return_type": "torchvision.datasets.SVHN",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.SVHN.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.UCF101"] = {
    "return_type": "torchvision.datasets.UCF101",
    "test_parameters": "('../data', frames_per_clip = 20, annotation_path = '../data/annfile.txt')",
}
allowlist["torchvision.datasets.UCF101.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.USPS"] = {
    "return_type": "torchvision.datasets.USPS",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.USPS.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.VOCSegmentation"] = {
    "return_type": "torchvision.datasets.VOCSegmentation",
    "test_parameters": "('../data',)",
}
allowlist["torchvision.datasets.VOCSegmentation.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.VOCDetection"] = {
    "return_type": "torchvision.datasets.VOCDetection",
    "test_parameters": "('../data',)",
}
allowlist[
    "torchvision.datasets.VOCDetection.__len__"
] = "torchvision.datasets.VOCDetection"

# Transforms
allowlist["torchvision.transforms.CenterCrop"] = {
    "return_type": "torchvision.transforms.CenterCrop",
    "test_parameters": "(10)",
}
allowlist["torchvision.transforms.ColorJitter"] = {
    "return_type": "torchvision.transforms.ColorJitter",
    "test_parameters": "(brightness=0, contrast=0, saturation=0, hue=0)",
}

# This is an interesting case, for some versions p = 0.2 is needed, for others its not needed

allowlist["torchvision.transforms.FiveCrop"] = {
    "return_type": "torchvision.transforms.FiveCrop",
    #    "test_parameters": "(size = 10, p = 0.2)",
}


allowlist["torchvision.transforms.Grayscale"] = {
    "return_type": "torchvision.transforms.Grayscale",
    "test_parameters": "(num_output_channels=1)",
}
allowlist["torchvision.transforms.Pad"] = {
    "return_type": "torchvision.transforms.Pad",
    "test_parameters": "(2, fill=0, padding_mode='constant')",
}
allowlist["torchvision.transforms.RandomAffine"] = {
    "return_type": "torchvision.transforms.RandomAffine",
    "test_parameters": "(degrees = 2)",
}

# transforms error
allowlist["torchvision.transforms.RandomApply"] = {
    "return_type": "torchvision.transforms.RandomApply",
    # "test_parameters": "(torchvision.transforms.CenterCrop(10))",
}

allowlist["torchvision.transforms.RandomCrop"] = {
    "return_type": "torchvision.transforms.RandomCrop",
    "test_parameters": "(size = 10)",
}
allowlist["torchvision.transforms.RandomGrayscale"] = {
    "return_type": "torchvision.transforms.RandomGrayscale",
    "test_parameters": "(p=0.1)",
}

allowlist["torchvision.transforms.RandomHorizontalFlip"] = {
    "return_type": "torchvision.transforms.RandomHorizontalFlip",
    "test_parameters": "(p=0.1)",
}
allowlist["torchvision.transforms.RandomPerspective"] = {
    "return_type": "torchvision.transforms.RandomPerspective",
    "test_parameters": "(distortion_scale=0.5, p=0.5)",
}

allowlist["torchvision.transforms.RandomResizedCrop"] = {
    "return_type": "torchvision.transforms.RandomResizedCrop",
    "test_parameters": "(10, scale=(0.08, 1.0), ratio=(0.75, 1.25))",
}
allowlist["torchvision.transforms.RandomRotation"] = {
    "return_type": "torchvision.transforms.RandomRotation",
    "test_parameters": "(degrees = 2)",
}
allowlist["torchvision.transforms.RandomSizedCrop"] = {
    "return_type": "torchvision.transforms.RandomSizedCrop",
    "test_parameters": "(10)",
}
allowlist["torchvision.transforms.RandomVerticalFlip"] = {
    "return_type": "torchvision.transforms.RandomVerticalFlip",
    "test_parameters": "(p=0.5)",
}
allowlist["torchvision.transforms.Resize"] = {
    "return_type": "torchvision.transforms.Resize",
    "test_parameters": "(size = 15)",
}
allowlist["torchvision.transforms.Scale"] = {
    "return_type": "torchvision.transforms.Scale",
    "test_parameters": "(10)",
}
allowlist["torchvision.transforms.TenCrop"] = {
    "return_type": "torchvision.transforms.TenCrop",
    "test_parameters": "(10)",
}
allowlist["torchvision.transforms.GaussianBlur"] = {
    "return_type": "torchvision.transforms.GaussianBlur",
    "min_version": "0.8.0",
    "test_parameters": "(kernel_size = 3)",
}

allowlist["torchvision.transforms.RandomChoice"] = {
    "return_type": "torchvision.transforms.RandomChoice",
}
allowlist["torchvision.transforms.RandomOrder"] = {
    "return_type": "torchvision.transforms.RandomOrder",
}

allowlist[
    "torchvision.transforms.LinearTransformation"
] = "torchvision.transforms.LinearTransformation"
allowlist[
    "torchvision.transforms.RandomErasing"
] = "torchvision.transforms.RandomErasing"
allowlist["torchvision.transforms.ConvertImageDtype"] = {
    "return_type": "torchvision.transforms.ConvertImageDtype",
    "min_version": "0.8.0",
}
allowlist["torchvision.transforms.ToPILImage"] = "torchvision.transforms.ToPILImage"
allowlist["torchvision.transforms.Lambda"] = "torchvision.transforms.Lambda"

# Functional Transformers
allowlist["torchvision.transforms.functional.adjust_brightness"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 0.5)",
}
allowlist["torchvision.transforms.functional.adjust_contrast"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 0.5)",
}
allowlist["torchvision.transforms.functional.adjust_gamma"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 1, 0.5)",
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.adjust_hue"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 0)"
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.adjust_saturation"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 0.5)",
}
allowlist["torchvision.transforms.functional.adjust_sharpness"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
    "test_parameters": "(tens, 0.5)",
}
allowlist["torchvision.transforms.functional.affine"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens,0.2,[1,2],0.2,[1,2])",
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.autocontrast"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
    "test_parameters": "(tens)",
}
allowlist["torchvision.transforms.functional.center_crop"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 10)",
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.convert_image_dtype"] = {
    "return_type": "torch.Tensor",
}
allowlist["torchvision.transforms.functional.crop"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 10 , 20, 30, 40)",
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.equalize"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
    "test_parameters": "(tens)",
}
allowlist["torchvision.transforms.functional.erase"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, 10, 20, 30, 40, 250)",
    "min_version": "0.8.0"
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.five_crop"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, 10)",
    "min_version": "0.8.0"
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.gaussian_blur"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens, 3)",
}
allowlist["torchvision.transforms.functional.hflip"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.invert"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
    "test_parameters": "(tens)",
}
allowlist["torchvision.transforms.functional.normalize"] = {
    "return_type": "torch.Tensor",
    # "test_parameters": "(tens, [0.5, 0.5, 0.5], [1, 1, 1]).unsqueeze(0)",
    # currently commenting because of test issues with hier versions
    # (//) works for 1.6.0 and / works for higher version :(
}
allowlist["torchvision.transforms.functional.pad"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, 10)",
    "min_version": "0.8.0"
    # # Torch 1.6 expects input to be PIL image, so commenting this currently
}
allowlist["torchvision.transforms.functional.perspective"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, [[10,20],[20,30],[30, 40],[40,50]], [[20,30],[30,40],[40, 50],[50,60]])",
    "min_version": "0.8.0"
    # # Torch 1.6 expects input to be PIL image, so commenting this currently
}

# Converts PIL to tensor, currently not supported
# allowlist["torchvision.transforms.functional.pil_to_tensor"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.posterize"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
    "test_parameters": "(tens, 2)",
}
allowlist["torchvision.transforms.functional.resize"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, 10)",
    "min_version": "0.8.0"
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.resized_crop"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, 10, 15, 20, 25, 30)",
    "min_version": "0.8.0"
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.rgb_to_grayscale"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
    "test_parameters": "(tens)",
}
allowlist["torchvision.transforms.functional.rotate"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, angle = 10)",
    "min_version": "0.8.0"
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.solarize"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
    "test_parameters": "(tens, threshold = 0.5)",
}
allowlist["torchvision.transforms.functional.ten_crop"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens, size = 10)",
    "min_version": "0.8.0"
    # Torch 1.6 expects input to be PIL image, so minimum version as 0.7 (Torch 1.7.0)
}
allowlist["torchvision.transforms.functional.to_grayscale"] = {
    "return_type": "PIL.Image.Image"
}
allowlist["torchvision.transforms.functional.to_pil_image"] = {
    "return_type": "PIL.Image.Image",
}
allowlist["torchvision.transforms.functional.to_tensor"] = {
    "return_type": "torch.Tensor",
}
allowlist["torchvision.transforms.functional.vflip"] = {
    "return_type": "torch.Tensor",
    "test_parameters": "(tens)",
}
