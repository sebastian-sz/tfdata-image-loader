# Further reading
This document presents further reading. If you are more interested in how
 `TFDataImageLoader` works you can find the detailed list of arguments below.  
 
This directory also contains 3 jupyter notebooks:
1. `quickstart.ipynb`
2. `benchmark.ipynb`
3. `extended_examples.ipynb`


## Arguments
Below is a detailed explanation of each possible argument you can pass to
 `TFDataImageLoader`:

|Argument | Type | Explanation|
---|---|---
|data_path| string|Path to the directory with folders containing images. Each directory name is considered to be a single class and it's expected to contain only images of that class.|
|target_size| tuple of two integers|The size for which to resize the loaded images.
|batch_size|int|How to batch both images and labels. For single image-label pair set this to 1.|
|shuffle*| boolean | Defaults to `True`. Whether to shuffle file names before loading.|
|cache| boolean| (optional) Defaults to `False`. Whether to cache dataset content. While giving some speed improvement, the default shuffling (if enabled) will happen only once. Also note that caching large dataset may fill your memory.
|mode| string | Defaults to `categorical`. Defines the type of labels. `sparse` makes the labels `0,1,2...`, while `categorical` will make it look it `[1, 0, 0,...], [0, 1, 0, 0...]`. This will throw error during creation of `TFDataImageLoader` if the mode is not one of the two above.
|pre_process_function**|a python function|(optional) Defaults to None. Knowing there are is a large variety of preprocessing types I left it to the user todefine his own preprocessing. The default loaded images are integers, in range `0-255`.
|augmentation_function***|a python function|(optional) Defaults to None. Similar structure to `pre_process_function`. The data that comes out of `pre_process_function` is passed to this function. Use it augment your data.
|verbose|boolean| Defaults to True. Whether to display loading information or not.
 _________________________  
 *shuffle  
 Note that I am not using `tf.data`'s `shuffle_buffer` as it's much slower and
  memory inefficient to shuffle image arrays rather than shuffle all file names before loading.
 
 If you still want `shuffle_buffer` style shuffling (for example after `cache()`) you
 can still call `.shuffle()` on returned dataset.  
 Example:
 ```python
train_loader = TFDataImageLoader(
    data_path="/home/user/data/dir",
    target_size=(224, 224),
    batch_size=8,
    cache=True,  # This is optional.
)

dataset = train_loader.load_dataset()
dataset = dataset.shuffle(shuffle_buffer=...)
```
_________________________

**pre_process_function  
This user defined function should take two arguments `image_batch, label_batch` and
 return the same
 arguments, but preprocessed.  
 All the operations performed there should be vectorised (they should work on a batch).   
 Lucky for us, basic math operations are pretty well supported for batch operations. Below is an example function, rescaling the images to 0-1 floats:
```python
 def my_pre_process(image_batch, label_batch):
    return image_batch / 255., label_batch
```
For mode advanced cases (regaridng `np.arrays`, not Graph Tensors) you could probably get away with Tensorflow's [numpy function](https://www.tensorflow.org/api_docs/python/tf/numpy_function).
_________________________
***augmentation_function:  
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
