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