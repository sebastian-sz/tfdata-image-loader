def empty_pre_process(image, label):
    return image, label


LOADER_CONFIG_FOR_TESTING = {
    "data_path": "./resources/images",
    "target_size": (240, 240),
    "pre_process_function": empty_pre_process,
    "shuffle": True,
    "batch_size": 1,
    "mode": "categorical",
    "verbose": 0,
}
