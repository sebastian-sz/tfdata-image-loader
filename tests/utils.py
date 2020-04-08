import hashlib
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from tests.config_for_testing import LOADER_CONFIG_FOR_TESTING

NUM_CHANNELS = 3


def calc_dataset_length(dataset: tf.data.Dataset) -> int:
    count = 0
    for _ in dataset:
        count += 1
    return count


def calculate_files_in_all_directories(directory: str) -> int:
    return sum([len(files) for _, _, files in os.walk(directory)])


def assert_dictionaries_the_same(some_dictionary: Dict, other_dictionary: Dict):
    assert _dictionaries_keys_equal(some_dictionary, other_dictionary)

    for key in some_dictionary:
        some_content = some_dictionary[key]
        other_content = other_dictionary[key]
        if isinstance(some_content, np.ndarray):
            assert (some_content == other_content).all()
        else:
            assert some_content == other_content


def _dictionaries_keys_equal(some_dictionary: Dict, other_dictionary: Dict):
    return sorted(some_dictionary.keys()) == sorted(other_dictionary.keys())


def stack_dataset_images(dataset: tf.data.Dataset) -> np.ndarray:
    return np.stack(
        [image.numpy()[0] for image, _ in dataset]
    )  # image array [img_count, h, w, c]


def list_dataset_labels(dataset: tf.data.Dataset) -> List[List[int]]:
    return [
        list(label.numpy()[0]) for _, label in dataset
    ]  # list of values like [0, 1]


def load_and_hash_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=NUM_CHANNELS)
    image = tf.image.resize(image, size=LOADER_CONFIG_FOR_TESTING["target_size"])
    return hash_image(image)


def hash_image(image: np.ndarray) -> str:
    return hashlib.sha224(image).hexdigest()


def empty_pre_process(
        image: tf.Tensor, label: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return image, label


def pre_process_for_0_1(
        image: tf.Tensor, label: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.divide(image, 255.0), label


def pre_process_for_minus_1_plus_1(
        image: tf.Tensor, label: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.divide(image, 127.5) - 1, label


def simple_augment(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.image.random_flip_left_right(image), label
