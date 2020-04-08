import pytest

from tests import utils
from tests.config_for_testing import LOADER_CONFIG_FOR_TESTING
from tfdata_image_loader import TFDataImageLoader


@pytest.fixture
def data_dir():
    return LOADER_CONFIG_FOR_TESTING["data_path"]


@pytest.fixture
def data_loader():
    return TFDataImageLoader(**LOADER_CONFIG_FOR_TESTING)


@pytest.fixture
def sparse_data_loader():
    sparse_loader_config = LOADER_CONFIG_FOR_TESTING.copy()
    sparse_loader_config["mode"] = "sparse"
    return TFDataImageLoader(**sparse_loader_config)


@pytest.fixture
def augmenting_data_loader():
    augmentations_present_config = LOADER_CONFIG_FOR_TESTING.copy()
    augmentations_present_config[
        "augmentation_function"
    ] = utils.simple_augment
    augmentations_present_config["shuffle"] = False
    return TFDataImageLoader(**augmentations_present_config)


@pytest.fixture
def data_loader_no_shuffling():
    no_shuffling_config = LOADER_CONFIG_FOR_TESTING.copy()
    no_shuffling_config["shuffle"] = False
    return TFDataImageLoader(**no_shuffling_config)
