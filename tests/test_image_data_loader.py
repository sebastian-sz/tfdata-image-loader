import os

import numpy as np
import pytest
import tensorflow as tf

from tests import utils
from tests.config_for_testing import LOADER_CONFIG_FOR_TESTING
from tfdata_image_loader import TFDataImageLoader


def test_loader_loading_all_images_in_subdirectories(data_loader, data_dir):
    total_images_count = utils.calculate_files_in_all_directories(data_dir)

    data_set = data_loader.load_dataset()  # Batch size 1
    loaded_images_count = utils.calc_dataset_length(data_set)

    assert total_images_count == loaded_images_count


def test_loader_taking_all_classes_into_account(data_dir, data_loader):
    class_directory_names = sorted(os.listdir(data_dir))

    assert (class_directory_names == data_loader.class_names).all()


def test_loader_correctly_creating_sparse_class_mapping(sparse_data_loader):
    expected_mapping = {0: "Class_1", 1: "Class_2"}

    assert expected_mapping == sparse_data_loader.class_names_mapping


def test_loader_correctly_making_categorical_class_mapping(data_loader):
    expected_mapping = {"Class_1": np.array([1, 0]), "Class_2": np.array([0, 1])}

    utils.assert_dictionaries_the_same(
        expected_mapping, data_loader.class_names_mapping
    )


@pytest.mark.parametrize("size_to_test", [(100, 100), (1920, 1080), (5, 200)])
def test_resizing_operations_work(size_to_test):
    loader_config = LOADER_CONFIG_FOR_TESTING.copy()
    loader_config["target_size"] = size_to_test
    data_loader = TFDataImageLoader(**loader_config)
    dataset = data_loader.load_dataset()

    expected_shape = (1, *size_to_test, 3)

    for image, _ in dataset:
        assert image.shape == expected_shape


@pytest.mark.parametrize(
    "function,min_value,max_value,expected_dtype",
    [
        (utils.pre_process_for_0_1, 0, 1, tf.float32),
        (utils.pre_process_for_minus_1_plus_1, -1, 1, tf.float32),
    ],
)
def test_pre_processing_being_correctly_applied(
    function, min_value, max_value, expected_dtype
):
    loader_config = LOADER_CONFIG_FOR_TESTING.copy()
    loader_config["pre_process_function"] = function
    data_loader = TFDataImageLoader(**loader_config)

    dataset = data_loader.load_dataset()

    for image, _ in dataset:
        assert image.dtype is expected_dtype
        assert np.min(image) >= min_value
        assert np.max(image) <= max_value


def test_if_shuffling_works(data_loader):
    max_runs = 5
    all_datasets_the_same = True  # Initial assumption

    dataset = data_loader.load_dataset()
    first_dataset_content = utils.list_dataset_labels(dataset)

    for _ in range(max_runs):
        another_dataset_content = utils.list_dataset_labels(dataset)

        if first_dataset_content != another_dataset_content:
            all_datasets_the_same = False
            break

    assert (
        not all_datasets_the_same,
        f"Shuffling didn't work. Out of all {max_runs} all datasets were the same.",
    )


def test_shuffling_can_be_disabled(data_loader_no_shuffling):
    max_runs = 5
    dataset = data_loader_no_shuffling.load_dataset()
    first_dataset_content = utils.list_dataset_labels(dataset)

    for _ in range(max_runs):
        another_dataset_content = utils.list_dataset_labels(dataset)

        if first_dataset_content != another_dataset_content:
            pytest.fail(
                "The data ordering is different despite shuffling set to False."
            )


def test_if_augmentation_functions_are_correctly_applied(augmenting_data_loader):
    max_runs = 5
    all_datasets_the_same = True

    dataset = augmenting_data_loader.load_dataset()
    first_dataset_content = utils.stack_dataset_images(dataset)

    for _ in range(max_runs):
        another_dataset_content = utils.stack_dataset_images(dataset)

        if (first_dataset_content != another_dataset_content).any():
            all_datasets_the_same = False
            break

    assert (
        not all_datasets_the_same,
        (
            f"The augmentations seem not to be applied, out of all {max_runs} loaded ",
            "datasets, all images were the same",
        ),
    )


def test_data_loader_correctly_matching_images_with_labels(data_loader):
    filename_label_mapping = {
        "./resources/images/Class_2/picture_1.jpg": np.array([0, 1]),
        "./resources/images/Class_2/picture_2.jpg": np.array([0, 1]),
        "./resources/images/Class_2/picture_3.jpg": np.array([0, 1]),
        "./resources/images/Class_2/picture_5.jpg": np.array([0, 1]),
        "./resources/images/Class_2/picture_6.jpg": np.array([0, 1]),
        "./resources/images/Class_2/picture_9.jpg": np.array([0, 1]),
        "./resources/images/Class_2/picture_10.jpg": np.array([0, 1]),
        "./resources/images/Class_1/picture_4.jpg": np.array([1, 0]),
        "./resources/images/Class_1/picture_7.jpg": np.array([1, 0]),
        "./resources/images/Class_1/picture_8.jpg": np.array([1, 0]),
    }

    expected_image_hash_label_mapping = {
        utils.load_and_hash_image(key): filename_label_mapping[key]
        for key in filename_label_mapping
    }

    dataset = data_loader.load_dataset()
    actual_hash_label_mapping = {
        utils.hash_image(image): label.numpy() for image, label in dataset
    }

    utils.assert_dictionaries_the_same(
        expected_image_hash_label_mapping, actual_hash_label_mapping
    )


def test_loader_correctly_calculating_the_number_of_dataset_steps(data_loader):
    dataset = data_loader.load_dataset()

    number_of_steps = utils.calc_dataset_length(dataset)

    assert data_loader.calc_expected_steps() == number_of_steps


def test_if_loader_correctly_calculates_image_count(data_loader, data_dir):
    images_number = utils.calculate_files_in_all_directories(data_dir)

    assert data_loader.get_image_count() == images_number


def test_loader_throwing_error_when_created_with_unsupported_mode(
    error_throwing_data_loader,
):
    data_loader, invalid_config = error_throwing_data_loader

    with pytest.raises(ValueError):
        data_loader(**invalid_config)
