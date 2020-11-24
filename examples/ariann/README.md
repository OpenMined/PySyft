# Installation
Define a `HOME` in data.py

You need to install 2 datasets in your `HOME`:
- Tiny Imagenet, from https://github.com/tjmoon0104/pytorch-tiny-imagenet
- Hymenoptera using the instructions:
    ```
    wget https://download.pytorch.org/tutorial/hymenoptera_data.zip\n"
                "unzip hymenoptera_data.zip
    ```
    
# Usage

```
python examples/ariann/main.py --model resnet18 --dataset hymenoptera --preprocess
```

# Datasets

## MNIST

1 x 28 x 28 pixel images

Suitability: Network1, Network2, LeNet

## CIFAR10
s
3 x 32 x32 pixel images

Suitability: AlexNet and VGG16

## Tiny Imagenet

3 x 64 x 64 pixel images

Suitability: AlexNet and VGG16

## Hymenoptera

3 x 64 x 64 pixel images

Suitability: ResNet18