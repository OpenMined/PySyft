# stdlib
from typing import Dict
from typing import Union

# transforms_ex = "transforms = torch.nn.Sequential(transforms.CenterCrop(10)"
# transforms_ex += ",transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)"
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
allowlist["torchvision.datasets.MNIST"] = "torchvision.datasets.MNIST"
allowlist["torchvision.datasets.MNIST.__len__"] = "syft.lib.python.Int"
allowlist["torchvision.datasets.VisionDataset"] = "torchvision.datasets.VisionDataset"
allowlist["torchvision.datasets.VisionDataset.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CelebA"] = "torchvision.datasets.CelebA"
allowlist["torchvision.datasets.CelebA.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CIFAR10"] = "torchvision.datasets.CIFAR10"
allowlist["torchvision.datasets.CIFAR10.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CIFAR100"] = "torchvision.datasets.CIFAR100"
allowlist["torchvision.datasets.CIFAR10.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Cityscapes"] = "torchvision.datasets.Cityscapes"
allowlist["torchvision.datasets.Cityscapes.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CocoCaptions"] = "torchvision.datasets.CocoCaptions"
allowlist["torchvision.datasets.CocoCaptions.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.CocoDetection"] = "torchvision.datasets.CocoDetection"
allowlist["torchvision.datasets.CocoDetection.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.EMNIST"] = "torchvision.datasets.EMNIST"
allowlist["torchvision.datasets.EMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.FakeData"] = "torchvision.datasets.FakeData"
allowlist["torchvision.datasets.FakeData.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.FashionMNIST"] = "torchvision.datasets.FashionMNIST"
allowlist["torchvision.datasets.FashionMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Flickr8k"] = "torchvision.datasets.Flickr8k"
allowlist["torchvision.datasets.Flickr8k.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Flickr30k"] = "torchvision.datasets.Flickr30k"
allowlist["torchvision.datasets.Flickr30k.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.HMDB51"] = "torchvision.datasets.HMDB51"
allowlist["torchvision.datasets.HMDB51.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.ImageNet"] = "torchvision.datasets.ImageNet"
allowlist["torchvision.datasets.ImageNet.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Kinetics400"] = "torchvision.datasets.Kinetics400"
allowlist["torchvision.datasets.Kinetics400.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.KMNIST"] = "torchvision.datasets.KMNIST"
allowlist["torchvision.datasets.KMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.LSUN"] = "torchvision.datasets.LSUN"
allowlist["torchvision.datasets.LSUN.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Omniglot"] = {
    "return_type": "torchvision.datasets.Omniglot",
    "min_version": "0.8.0",
}
allowlist["torchvision.datasets.Omniglot.__len__"] = {
    "return_type": "syft.lib.python.Int",
    "min_version": "0.8.0",
}

allowlist["torchvision.datasets.PhotoTour"] = "torchvision.datasets.PhotoTour"
allowlist["torchvision.datasets.PhotoTour.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.Places365"] = {
    "return_type": "torchvision.datasets.Places365",
    "min_version": "0.8.0",
}
allowlist["torchvision.datasets.Places365.__len__"] = {
    "return_type": "syft.lib.python.Int",
    "min_version": "0.8.0",
}

allowlist["torchvision.datasets.QMNIST"] = "torchvision.datasets.QMNIST"
allowlist["torchvision.datasets.QMNIST.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.SBDataset"] = "torchvision.datasets.SBDataset"
allowlist["torchvision.datasets.SBDataset.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.SBU"] = "torchvision.datasets.SBU"
allowlist["torchvision.datasets.SBU.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.STL10"] = "torchvision.datasets.STL10"
allowlist["torchvision.datasets.STL10.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.SVHN"] = "torchvision.datasets.SVHN"
allowlist["torchvision.datasets.SVHN.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.UCF101"] = "torchvision.datasets.UCF101"
allowlist["torchvision.datasets.UCF101.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.USPS"] = "torchvision.datasets.USPS"
allowlist["torchvision.datasets.USPS.__len__"] = "syft.lib.python.Int"

allowlist[
    "torchvision.datasets.VOCSegmentation"
] = "torchvision.datasets.VOCSegmentation"
allowlist["torchvision.datasets.VOCSegmentation.__len__"] = "syft.lib.python.Int"

allowlist["torchvision.datasets.VOCDetection"] = "torchvision.datasets.VOCDetection"
allowlist[
    "torchvision.datasets.VOCDetection.__len__"
] = "torchvision.datasets.VOCDetection"

allowlist["torchvision.transforms.CenterCrop"] = {
    "return_type": "torchvision.transforms.CenterCrop",
    "test_parameters": "(10)",
}
allowlist["torchvision.transforms.ColorJitter"] = {
    "return_type": "torchvision.transforms.ColorJitter",
    "test_parameters": "(brightness=0, contrast=0, saturation=0, hue=0)",
}

# This is an interesting case, for some versions p = 0.2 is needed, for others its not needed

# allowlist["torchvision.transforms.FiveCrop"] = {
#    "return_type": "torchvision.transforms.FiveCrop",
#    "test_parameters": "(size = 10, p = 0.2)",
# }


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

# allowlist["torchvision.transforms.RandomApply"] = {
#    "return_type": "torchvision.transforms.RandomApply",
#    "test_parameters": "(torchvision.transforms.CenterCrop(10))",
# }

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
allowlist["torchvision.transforms.Normalize"] = "torchvision.transforms.Normalize"
allowlist[
    "torchvision.transforms.RandomErasing"
] = "torchvision.transforms.RandomErasing"
allowlist["torchvision.transforms.ConvertImageDtype"] = {
    "return_type": "torchvision.transforms.ConvertImageDtype",
    "min_version": "0.8.0",
}
allowlist["torchvision.transforms.ToPILImage"] = "torchvision.transforms.ToPILImage"
allowlist["torchvision.transforms.Lambda"] = "torchvision.transforms.Lambda"

allowlist["torchvision.transforms.functional.adjust_brightness"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.adjust_contrast"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
}
allowlist["torchvision.transforms.functional.adjust_gamma"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.adjust_hue"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.adjust_saturation"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.adjust_sharpness"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
}
allowlist["torchvision.transforms.functional.affine"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.autocontrast"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
}
allowlist["torchvision.transforms.functional.center_crop"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.convert_image_dtype"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.crop"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.equalize"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
}
allowlist["torchvision.transforms.functional.erase"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.five_crop"] = "torch.Tensor"
allowlist["orchvision.transforms.functional.gaussian_blur"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
}
allowlist["torchvision.transforms.functional.hflip"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.invert"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
}
allowlist["torchvision.transforms.functional.normalize"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.pad"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.perspective"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.pil_to_tensor"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.posterize"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
}
allowlist["torchvision.transforms.functional.resize"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.resized_crop"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.rgb_to_grayscale"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.8.0",
}
allowlist["torchvision.transforms.functional.rotate"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.solarize"] = {
    "return_type": "torch.Tensor",
    "min_version": "0.9.0",
}
allowlist["torchvision.transforms.functional.ten_crop"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.to_grayscale"] = "torch.Tensor"

# allowlist["torchvision.transforms.functional.to_pil_image"] = "PIL.Image.Image"

allowlist[" torchvision.transforms.functional.to_tensor"] = "torch.Tensor"
allowlist["torchvision.transforms.functional.vflip"] = "torch.Tensor"
