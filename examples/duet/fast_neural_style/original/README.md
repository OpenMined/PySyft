# fast_neural_style :city_sunrise: :rocket:
This repository contains a PyTorch implementation of an algorithm for artistic style transfer. The algorithm can be used to mix the content of an image with the style of another image. For example, here is a photograph of a door arch rendered in the style of a stained glass painting.

The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf). The saved-models for examples shown in the README can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0).

<p align="center">
    <img src="images/style-images/mosaic.jpg" height="200px">
    <img src="images/content-images/amber.jpg" height="200px">
    <img src="images/output-images/amber-mosaic.jpg" height="440px">
</p>

## Requirements
The program is written in Python, and uses [pytorch](http://pytorch.org/), [scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop using saved models.

## Usage
Stylize image
```bash
python neural_style/neural_style.py eval --model </path/to/saved/model> --content_image </path/to/content/image> --output_image </path/to/output/image> --cuda 0
```
* `--content_image`: path to content image you want to stylize.
* `--model`: saved model to be used for stylizing the image (eg: `mosaic.pth`)
* `--output_image`: path for saving the output image.
* `--content_scale`: factor for scaling down the content image if memory is an issue (eg: value of 2 will halve the height and width of content-image)
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Train model
```bash
python neural_style/neural_style.py train --dataset </path/to/train-dataset> --style_image </path/to/style/image> --save_model_dir </path/to/save-model/folder> --epochs 2 --cuda 1
```

There are several command line arguments, the important ones are listed below
* `--dataset`: path to training dataset, the path should point to a folder containing another folder with all the training images. I used COCO 2014 Training images dataset [80K/13GB] [(download)](http://mscoco.org/dataset/#download).
* `--style_image`: path to style-image.
* `--save_model_dir`: path to folder where trained model will be saved.
* `--cuda`: set it to 1 for running on GPU, 0 for CPU.

Refer to ``neural_style/neural_style.py`` for other command line arguments. For training new models you might have to tune the values of `--content_weight` and `--style_weight`. The mosaic style model shown above was trained with `--content_weight 1e5` and `--style_weight 1e10`. The remaining 3 models were also trained with similar order of weight parameters with slight variation in the `--style_weight` (`5e10` or `1e11`).

## Models

Models for the examples shown below can be downloaded from [here](https://www.dropbox.com/s/lrvwfehqdcxoza8/saved_models.zip?dl=0) or by running the script ``download_saved_models.py``.

<div align='center'>
  <img src='images/content-images/amber.jpg' height="174px">
</div>

<div align='center'>
  <img src='images/style_images/mosaic.jpg' height="174px">
  <img src='images/output_images/amber_mosaic.jpg' height="174px">
  <img src='images/output_images/amber_candy.jpg' height="174px">
  <img src='images/style_images/candy.jpg' height="174px">
  <br>
  <img src='images/style_images/rain_princess_cropped.jpg' height="174px">
  <img src='images/output_images/amber_rain_princess.jpg' height="174px">
  <img src='images/output_images/amber_udnie.jpg' height="174px">
  <img src='images/style_images/udnie.jpg' height="174px">
</div>
