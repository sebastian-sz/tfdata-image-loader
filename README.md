## Table of contents
  
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic usage](#basic-usage)
4. [Performance](#performance)
5. [Advanced usage](#advanced-usage)
6. [Afterword](#afterword)

## Introduction
I am sharing my code for loading Image Data for Image Classification Problem using Tensorflow `tf.data.Dataset`.  

For basic usage, you are supposed to:
 * pass a path to a directory containing directories with images. It's expected that 
every directory contains images from a single class.
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

##### Module installation:
Clone manually:
```bash
git clone https://github.com/sebastian-sz/tfdata-image-loader.git   
cd tfdata-image-loader
pip install -e .
```
After that, you can simply call:  
`from tfdata_image_loader import TFDataImageLoader`

##### Running tests
I'm using [pytest](https://docs.pytest.org/en/latest/) library for testing. To run the tests, please install the necessary testing dependencies
and run `pytest` in your terminal from the `tests` directory:
```
cd tfdata-image-loader/tests
pip install -r requirements.txt
pytest
```

## Basic usage
Here is a minimal example on how to load an image-label `tf.data.Dataset` using `tfdata-image-loader`:
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

For detailed explanation of the flow and arguments, please refer to Section 5: Advanced Usage.

## Performance
One of the key motivations behind `tfdata-image-loader` was increased performance in data loading. 
Following Tensorflow's best practices I am using `tf.data.Dataset` API to load and pre-process data.
  
In the [Benchmark Colab Notebook](https://colab.research.google.com/drive/1iHQr9nW8g8oCfgvML0e_Ay4TadekqyHw) I am comparing default Keras `flow_from_directory` with my 
implementation, which results in ~1.5x training speedup when doing basic fine tuning. 

## Advanced usage
Below is a detailed explanation of each possible argument you can pass to `TFDataLoader`:
##### data_path: `string`
   
 Path to the directory with folders containing images. Each directory name is considered to be a single class.
            
 ##### target_size: (`int, int`) 
 The size for which to resize the loaded images
 ##### batch_size: `int`
 How to batch both images and labels. For single image-label pair set this to 1.
 
 ##### shuffle: `bool` (optional)
 Defaults to `True`. Whether to shuffle file names before loading.   
 
 Note that I am not using `tf.data`'s `shuffle_buffer` as it's much slower and memory inefficient to shuffle image arrays rather than shuffle all file names before loading.
 (This might be changed in the future)  
 
 ##### cache : `bool` (optional) 
 Defaults to `False`.  
 Whether to cache dataset content. While giving some speed improvement, the shuffling (if enabled) will happen only once.
 Also note that caching large dataset may fill your memory.
 
 ##### mode:`string` (optional)
  `"categorical"` or `"sparse"`.   
  Defines the type of labels. `sparse` makes the labels 0,1,2... categorical` will make it look it [1, 0, 0,...], [0, 1, 0, 0...].
  
 ##### pre_process_function: a python function (optional) 
 Knowing there are is a large variety of preprocessing types I left it to the user to define his own preprocessing. The loaded images are integers, in rage 0-255.  
   
 `pre_process_function` should take two arguments `image_batch, label_batch` and return the same arguments, but preprocessed.  
 All the operations performed there should be vectorised (they should work on a batch).   
 Lucky for us, basic math operations are pretty well supported for batch operations. Below is an example function, rescaling the images to 0-1 floats:
```python
 def my_pre_process(image_batch, label_batch):
    return image_batch / 255., label_batch
```
For mode advanced cases (regaridng `np.arrays`, not Graph Tensors) you could probably get away with Tensorflow's [numpy function](https://www.tensorflow.org/api_docs/python/tf/numpy_function).

##### augmentation_function: a python function (optional) 
Similar to `pre_process_function`. The function should be vectorised. and accept `image_batch, label_batch`. 
It should return `image_batch, label_batch` too.  
I recommend using `tf.image.random(...)` operations, which seem to be most suitable for this implementation.  
In the example below I'm writing a function that will randomly flip some examples in the image batch:
```python
def my_augment(image_batch, label_batch):
    flipped_image_batch = tf.image.random_flip_left_right(image_batch) 
    return flipped_image_batch, label
```
For mode advanced cases (regaridng `np.arrays`, not Graph Tensors), you could probably get away with Tensorflow's [numpy function](https://www.tensorflow.org/api_docs/python/tf/numpy_function).
                
##### verbose:`bool` (optional)
Defaults to True.  Whether to display loading information or not.
  
## Afterword
The `tfdata-image-loader` module was written by following simplicity and minimalism principles, while still leaving much 
freedom to the potential end users and ensuring satisfactory performance.  
If you have questions or ideas on how to improve this package, let me know!
