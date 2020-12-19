# Super-Resolution Using An Efficient Sub-Pixel Convolutional Neural Network

This example illustrates how to use the efficient sub-pixel convolution layer described in [Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution within your network for tasks such as super-resolution.

```bash
usage: main.py [-h] --upscale-factor UPSCALE_FACTOR [--batch-size BATCHSIZE]
               [--test-batch-size TESTBATCHSIZE] [--epochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --upscale-factor      super resolution upscale factor
  --batch-size          training batch size
  --test-batch-size     testing batch size
  --epochs              number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
```
This example trains a super-resolution network on the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. A snapshot of the model after every epoch with filename model_epoch_<epoch_number>.pth

## Example Usage:

### Train

```bash
python main.py --upscale-factor 3 --batch-size 4 --test-batch-size 100 --epochs 30 --lr 0.001
```

### Super Resolve
```bash
python super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --model model_epoch_500.pth --output_filename out.png
```
