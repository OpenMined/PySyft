# Deep Convolution Generative Adversarial Networks

This example implements the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](http://arxiv.org/abs/1511.06434)

The implementation is very close to the Torch implementation [dcgan.torch](https://github.com/soumith/dcgan.torch)

After every 100 training iterations, the files `real_samples.png` and `fake_samples.png` are written to disk
with the samples from the generative model.

After every epoch, models are saved to: `net_generator_epoch_%d.pth` and `net_discriminator_epoch_%d.pth`

## Downloading the dataset
You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun), and running
```bash
python download.py -c bedroom
```

## Usage
```bash
usage: main.py [-h] --dataset DATASET --data_folder DATAFOLDER [--workers WORKERS]
               [--batch_size BATCHSIZE] [--image_size IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--n_iter NITER] [--lr LR]
               [--beta1 BETA1] [--cuda] [--n_gpu NGPU] [--net_generator NET_GENERATOR]
               [--net_discriminator NET_DISCRIMINATOR] [--out_folder OUT_FOLDER]
               [--manual_seed SEED] [--classes CLASSES]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | mnist |imagenet | folder | lfw | fake
  --data_folder DATAFOLDER   path to dataset
  --workers WORKERS     number of data loading workers
  --batch_size BATCHSIZE input batch size
  --image_size IMAGESIZE the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --n_iter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --n_gpu NGPU           number of GPUs to use
  --net_generator NET_GENERATOR           path to net_generator (to continue training)
  --net_discriminator NET_DISCRIMINATOR           path to net_discriminator (to continue training)
  --out_folder OUTF           folder to output images and model checkpoints
  --manual_seed SEED     manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set
```
