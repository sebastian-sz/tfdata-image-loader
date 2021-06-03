import hashlib
import os
import unittest

import numpy as np
import tensorflow as tf

from tests._root_dir import ROOT_DIR
from tfdata_image_loader import TFDataImageLoader


class TestImageDataLoader(unittest.TestCase):
    config = {
        "data_path": os.path.join(ROOT_DIR, "resources/images"),
        "target_size": (240, 240),
        "pre_process_function": lambda x, y: (x, y),
        "shuffle": True,
        "batch_size": 1,
        "mode": "categorical",
        "verbose": 0,
    }

    loader = TFDataImageLoader(**config)

    def test_loader_loading_all_images_in_subdirectories(self):
        expected_images_count = 10  # content of resources/images

        dataset = self.loader.load_dataset()  # Batch size 1
        num_elements = sum([1 for _ in dataset])

        assert expected_images_count == num_elements

    def test_loader_taking_all_classes_into_the_account(self):
        data_dir = self.config["data_path"]
        class_directory_names = sorted(os.listdir(data_dir))

        assert (class_directory_names == self.loader.class_names).all()

    def test_categorical_class_mapping(self):
        expected_mapping = {"Class_1": np.array([1, 0]), "Class_2": np.array([0, 1])}

        assert expected_mapping.keys() == self.loader.class_names_mapping.keys()

        for class_key, expected_value in expected_mapping.items():
            actual_value = self.loader.class_names_mapping[class_key]
            assert (expected_value == actual_value).all()

    def test_sparse_class_mapping(self):
        expected_mapping = {0: "Class_1", 1: "Class_2"}

        config = self.config.copy()
        config["mode"] = "sparse"
        loader = TFDataImageLoader(**config)

        assert expected_mapping == loader.class_names_mapping

    def test_resizing_operations_work(self):
        sizes_to_test = [(100, 100), (1920, 1080), (5, 200)]
        for size_to_test in sizes_to_test:
            self._test_resize(size_to_test)

    def _test_resize(self, size_to_test):
        config = self.config.copy()
        config["target_size"] = size_to_test
        data_loader = TFDataImageLoader(**config)
        dataset = data_loader.load_dataset()

        expected_shape = (1, *size_to_test, 3)

        for image, _ in dataset:
            assert image.shape == expected_shape

    def test_preprocessing_types(self):
        preprocessing_params = [
            {  # Preprocess from 0-255 to 0-1
                "function": lambda image, label: (image / 255.0, label),
                "min_val": 0,
                "max_val": 1,
                "dtype": tf.float32,
            },
            {  # Preprocess from 0-255 to -1+1
                "function": lambda image, label: ((image / 127.5) - 1, label),
                "min_val": -1,
                "max_val": 1,
                "dtype": tf.float32,
            },
        ]

        for preprocessing_type in preprocessing_params:
            self._test_preprocessing(preprocessing_type)

    def _test_preprocessing(self, preprocessing_type):
        config = self.config.copy()
        config["pre_process_function"] = preprocessing_type["function"]
        data_loader = TFDataImageLoader(**config)

        dataset = data_loader.load_dataset()

        for image, _ in dataset:
            assert image.dtype is preprocessing_type["dtype"]
            assert np.min(image) >= preprocessing_type["min_val"]
            assert np.max(image) <= preprocessing_type["max_val"]

    def test_shuffling_works(self):
        dataset = self.loader.load_dataset()

        first_run = np.array([image.numpy() for image, _ in dataset])
        second_run = np.array([image.numpy() for image, _ in dataset])

        assert not np.allclose(first_run, second_run)

    def test_shuffling_can_be_disabled(self):
        config = self.config.copy()
        config["shuffle"] = False
        loader = TFDataImageLoader(**config)
        dataset = loader.load_dataset()

        first_run = np.array([image.numpy() for image, _ in dataset])
        second_run = np.array([image.numpy() for image, _ in dataset])

        assert np.allclose(first_run, second_run)

    def test_augmentations(self):
        # Capture not shuffled, not augmented data:
        deterministic_config = self.config.copy()
        deterministic_config["shuffle"] = False
        loader = TFDataImageLoader(**deterministic_config)
        dataset = loader.load_dataset()
        not_augmented_data = np.array([image.numpy() for image, _ in dataset])

        # Capture not shuffled, augmented data:
        config = self.config.copy()
        config["augmentation_function"] = lambda image, label: (
            tf.image.random_flip_left_right(image),
            label,
        )
        config["shuffle"] = False
        loader = TFDataImageLoader(**config)
        dataset = loader.load_dataset()
        augmented_data = np.array([image.numpy() for image, _ in dataset])

        assert not np.allclose(not_augmented_data, augmented_data)

    def test_loader_estimating_dataset_steps(self):
        dataset = self.loader.load_dataset()
        num_steps = sum([1 for _ in dataset])

        assert self.loader.calc_expected_steps() == num_steps

    def test_loader_estimating_image_count(self):
        expected_image_number = 10  # Content of resources/images

        assert self.loader.get_image_count() == expected_image_number

    def test_loader_matching_image_with_label(self):
        filenames = [
            # Class 1
            "resources/images/Class_2/picture_1.jpg",
            "resources/images/Class_2/picture_2.jpg",
            "resources/images/Class_2/picture_3.jpg",
            "resources/images/Class_2/picture_5.jpg",
            "resources/images/Class_2/picture_6.jpg",
            "resources/images/Class_2/picture_9.jpg",
            "resources/images/Class_2/picture_10.jpg",
            # Class 2
            "resources/images/Class_1/picture_4.jpg",
            "resources/images/Class_1/picture_7.jpg",
            "resources/images/Class_1/picture_8.jpg",
        ]

        labels = [
            # Class 1
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0, 1]),
            # Class 2
            np.array([1, 0]),
            np.array([1, 0]),
            np.array([1, 0]),
        ]

        image_hash_label_map = self._make_image_hash_label_map(filenames, labels)

        dataset = self.loader.load_dataset()

        for image, label in dataset:
            image_hash = self._hash_image(image.numpy())
            expected_label = image_hash_label_map[image_hash]

            assert np.equal(expected_label, label.numpy()).all()

    def _make_image_hash_label_map(self, filenames, labels):
        full_filenames = [os.path.join(ROOT_DIR, x) for x in filenames]

        image_hashes = []
        for filename in full_filenames:
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, size=self.config["target_size"])
            image_hash = self._hash_image(image.numpy())
            image_hashes.append(image_hash)

        return {image_hash: label for image_hash, label in zip(image_hashes, labels)}

    @staticmethod
    def _hash_image(image):
        return hashlib.sha224(image).hexdigest()

    def test_loader_raising_error_with_unsupported_class_mode(self):
        config = self.config.copy()
        config["mode"] = "unsupported mode"

        with self.assertRaises(ValueError):
            TFDataImageLoader(**config)


if __name__ == "__main__":
    unittest.main()
