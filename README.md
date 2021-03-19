## Table of contents
  
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic usage](#basic-usage)
4. [Performance](#performance)
5. [Afterword](#afterword)

## Introduction
I am sharing my code for loading Image Data for Image Classification Problem using Tensorflow's `tf.data.Dataset`.  

For basic usage:
 * pass a path to a directory containing directories with images (it's expected that 
every directory contains images from a single class).
 * define: batch size, target image size and pre processing function.    

##### Motivation
When designing this module I wanted to load images faster than Keras `flow_from_directory`, preferably 
using `tf.data`, while still keeping a lot of flexibility.

##### Quick Start (Google Colab) 
I mostly designed this to be used inside Google Colab but you are free to use it anywhere else.  
This [Colab Quickstart Example](https://colab.research.google.com/drive/11Qpe8zJB6qjO4oAqQASzDHe0r0yP9sok) shows a minimal example on how to install and use this mini-package.

## Installation
##### Requierments:
 * Python 3.6+
 * Tensorflow 2.x

##### Pip one-liner:
`pip install -e git+git://github.com/sebastian-sz/tfdata-image-loader.git#egg=tfdata-image-loader`

##### (Alternatively) Build from source:
```bash
git clone https://github.com/sebastian-sz/tfdata-image-loader.git   
cd tfdata-image-loader
pip install -e .
```
Either way, if all goes well, you should be able to simply call:  
`from tfdata_image_loader import TFDataImageLoader`

##### Running tests
I'm using built-in `unittest` library for testing. To run the tests run `make test` in
your terminal.   
(Alternatively, `python -m unittest discover tests/`)

## Basic usage
Here is a minimal example on how to create an image-label pairs `tf.data.Dataset` using `tfdata-image-loader`.  
I'm going to resize to (224,224), rescale images to 0-1 and randomly flip some images.

```python
import tensorflow as tf

from tfdata_image_loader import TFDataImageLoader


def my_preprocess(image, label):
    return image / 255., label


def my_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


train_loader = TFDataImageLoader(
    data_path="/home/user/path/to/your/data",
    target_size=(224, 224),
    batch_size=32,
    pre_process_function=my_preprocess,
    augmentation_function=my_augment,
)

train_dataset = train_loader.load_dataset()

model = (...)
model.fit(train_dataset, ...)
```
Check out [docs/](https://github.com/sebastian-sz/tfdata-image-loader/tree/master/docs
) for detailed explanation of possible arguments, and examples of processing raw numpy
 arrays.


## Performance
One of the key motivations behind `tfdata-image-loader` was increased performance in data loading.
Below is a snippet from my benchmark* comparing Tensorflow's image loading techniques:
1. `tfdata-image-loader` (this repo): `63 ms/step`
2. Keras `ImageDataGenerator`: `121 ms/step`
3. (New in TF 2.3) keras.preprocessing: `image_dataset_from_directory`: `95 ms/step` 

*Benchmark notebook is available [here](https://colab.research.google.com/drive/1tsVqYcb_FE5pfAilG8Bybzbqkl5Mo5Zn?usp=sharing).  
  
## Afterword
The `tfdata-image-loader` module was written by following simplicity and minimalism principles, while still leaving much 
freedom to the potential end users and ensuring satisfactory performance.  
If you have questions or ideas on how to improve this package, let me know!
