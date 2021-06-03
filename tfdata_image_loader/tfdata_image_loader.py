"""A class to load image, label pairs as a Tensorflow Dataset.

Provides tf.data.Dataset object that can be used to train and evaluate deep learning
models (mostly for Image Classification).
"""

import os
import pathlib
from typing import Callable, Dict

import numpy as np
import tensorflow as tf


class TFDataImageLoader:
    """
    Loads image, label data pairs using tensorflow's tf.data API.

    The returned data is loaded based on directory structure. It is assumed that the
    directory contains folders that represent classes.
    The order of the operations is as follows:
    1. load filenames.
    2. load labels.
    3. load and resize images.
    4. (optional) pre process data.
    5. (optional) augment data.
    6. batch images and labels.
    7. (optional) cache dataset.
    8. prefetch more samples in advance.

    Arguments:
        data path: path to the folder containing directories, containing images.
        target_size: the size for which to resize the loaded images.
        batch_size: number of image, label pairs in a single batch in the dataset.
        shuffle: whether to shuffle filenames while reading files from disk. Defaults to
            True. If you want to shuffle the end dataset it is advised to use .shuffle.
        cache: whether to cache the preprocessed and augmented dataset in memory.
            Defaults to False.
        mode: either "categorical" or "sparse". Defaults to categorical. "categorical"
            will load labels as [0, 0, 1, 0, ...] whereas "sparse" will load them as
            1,2,3 integer. This will raise ValueError if you pass invalid value.
        pre_process_function: a function accepting and returning image, label pair.
            The preprocessing should happen inside this function. Defaults to None.
            The function should be vectorized and support tf.data.Dataset map operation.
            If not provided the images will be loaded as 0-255 integers.
        augmentation_function: a function accepting and returning image, label pair.
            Defaults to None. Data augmentations should happen in this function. The
            function should be vectorized and support tf.data.Dataset map operation.
            If not provided the data will be loaded as it comes out of
            pre_process_function.
        verbose: Whether to display data information upon creation. Defaults to True.

    Example:
        data_loader = TFDataImageLoader(
            data_path="./data",
            target_size=(224, 224),
            batch_size=8
            )
        dataset = data_loader.load_dataset()
        for image, label in dataset:
            ...

    Raises:
        ValueError if provided mode is not one of "categorical" or "sparse".
    """

    NUM_CHANNELS = 3
    AUTO_TUNE = tf.data.experimental.AUTOTUNE

    def __init__(
        self,
        data_path: str,
        target_size: (int, int),
        batch_size: int,
        shuffle: bool = True,
        cache: bool = False,
        mode: str = "categorical",
        pre_process_function: Callable = None,
        augmentation_function: Callable = None,
        verbose: bool = True,
    ):
        self.data_dir = pathlib.Path(data_path)
        self.target_size = target_size
        self.batch_size = batch_size
        self.pre_process_function = pre_process_function
        self.augmentation_function = augmentation_function
        self.shuffle = shuffle
        self.cache = cache
        self.mode = mode

        self.class_names = self._get_class_names()
        self.class_names_mapping = self._get_label_mapping()

        if verbose:
            self._print_message()

    def _get_class_names(self) -> np.ndarray:
        return np.sort(np.array([item.name for item in self.data_dir.glob("*")]))

    def _get_label_mapping(self) -> Dict:
        if self.mode == "sparse":
            return {value: name for value, name in enumerate(self.class_names)}
        elif self.mode == "categorical":
            return {
                name: (name == self.class_names).astype(np.int32)
                for name in self.class_names
            }
        else:
            raise ValueError(
                f"Unsupported mode type: {self.mode}. Available modes"
                f"are 'categorical' or 'sparse'."
            )

    def _print_message(self):
        message = (
            f"Found {self.get_image_count()} images, "
            f"belonging to {len(self.class_names)} classes"
        )
        print(message)
        print("")
        print("Class names mapping: ")
        print(self.class_names_mapping)
        print("")

    def get_image_count(self):
        """Return the number of images detected in the data directory."""
        return len(list(self.data_dir.glob("*/*")))

    def load_dataset(self):
        """Return tf.data.Dataset based on parameters passed to the constructor."""
        file_names_dataset = self._load_file_names()
        return (
            file_names_dataset.map(
                self._load_img_and_label, num_parallel_calls=self.AUTO_TUNE
            )
            .apply(self._maybe_apply_pre_processing)
            .apply(self._maybe_apply_augmentation)
            .batch(self.batch_size)
            .apply(self._maybe_cache_content)
            .prefetch(self.AUTO_TUNE)
        )

    def _load_file_names(self):
        return tf.data.Dataset.list_files(
            str(self.data_dir / "*/*"), shuffle=self.shuffle
        )

    def _load_img_and_label(self, file_path):
        label = self._get_label(file_path)
        image = self._get_image(file_path)
        return tf.image.resize(image, self.target_size), label

    def _get_label(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        bool_label = parts[-2] == self.class_names
        categorical_int_label = tf.cast(bool_label, tf.int32)

        if self.mode == "sparse":
            return tf.argmax(categorical_int_label)

        return categorical_int_label

    def _get_image(self, file_path):
        image = tf.io.read_file(file_path)
        return tf.image.decode_jpeg(image, channels=self.NUM_CHANNELS)

    def _resize_images(self, image, label):
        return tf.image.resize(image, self.target_size), label

    def _maybe_apply_pre_processing(self, dataset):
        if self.pre_process_function:
            dataset = dataset.map(
                self.pre_process_function, num_parallel_calls=self.AUTO_TUNE
            )
        return dataset

    def _maybe_apply_augmentation(self, dataset):
        if self.augmentation_function:
            dataset = dataset.map(
                self.augmentation_function, num_parallel_calls=self.AUTO_TUNE
            )
        return dataset

    def _maybe_cache_content(self, dataset):
        if self.cache:
            dataset = dataset.cache()
        return dataset

    def calc_expected_steps(self):
        """Return the expected number of steps in one dataset loop."""
        return int(self.get_image_count() / self.batch_size)
