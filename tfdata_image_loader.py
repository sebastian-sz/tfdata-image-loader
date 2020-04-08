import os
import pathlib
from typing import Dict

import numpy as np
import tensorflow as tf


class TFDataImageLoader:

    NUM_CHANNELS = 3
    AUTO_TUNE = tf.data.experimental.AUTOTUNE

    def __init__(
        self,
        data_path: str,
        target_size: (int, int),
        batch_size: int,
        shuffle=True,
        cache=False,
        mode="categorical",
        pre_process_function=None,
        augmentation_function=None,
        verbose=True,
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
        return np.sort(np.array([item.name for item in self.data_dir.glob('*')]))

    def _get_label_mapping(self) -> Dict:
        if self.mode == "sparse":
            return {value: name for value, name in enumerate(self.class_names)}
        return {
            name: (name == self.class_names).astype(np.int32)
            for name in self.class_names
        }

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
        return len(list(self.data_dir.glob('*/*')))

    def load_dataset(self):
        file_names_dataset = self._load_file_names()
        return (
            file_names_dataset
            .map(self._load_img_and_label, num_parallel_calls=self.AUTO_TUNE)
            .batch(self.batch_size)
            .apply(self._maybe_apply_pre_processing)
            .apply(self._maybe_apply_augmentation)
            .apply(self._maybe_cache_content)
            .prefetch(self.AUTO_TUNE)
        )

    def _load_file_names(self):
        return tf.data.Dataset.list_files(
            str(self.data_dir / '*/*'), shuffle=self.shuffle
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
                self.pre_process_function, num_parallel_calls=self.AUTO_TUNE,
            )
        return dataset

    def _maybe_apply_augmentation(self, dataset):
        if self.augmentation_function:
            dataset = dataset.map(
                self.augmentation_function, num_parallel_calls=self.AUTO_TUNE,
            )
        return dataset

    def _maybe_cache_content(self, dataset):
        if self.cache:
            dataset = dataset.cache()
        return dataset

    def calc_expected_steps(self):
        """
        Returns the expected number of steps in one dataset loop.
        """
        return int(self.get_image_count() / self.batch_size)
