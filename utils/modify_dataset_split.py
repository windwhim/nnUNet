import json

import os

path_map = {
    "001": {
        "train": {
            "image": "/root/autodl-tmp/dataset/frame/train/image",
            "mask": "/root/autodl-tmp/dataset/frame/train/mask/frame",
        },
        "test": {
            "image": "/root/autodl-tmp/dataset/frame/test/image",
            "mask": "/root/autodl-tmp/dataset/frame/test/mask/frame",
        },
    },
    "002": {
        "train": {
            "image": "/root/autodl-tmp/dataset/frame/train/image",
            "mask": "/root/autodl-tmp/dataset/frame/train/mask/lens",
        },
        "test": {
            "image": "/root/autodl-tmp/dataset/frame/test/image",
            "mask": "/root/autodl-tmp/dataset/frame/test/mask/lens",
        },
    },
    "101": {
        "train": {
            "image": "/root/autodl-tmp/dataset/noframe/train/image",
            "mask": "/root/autodl-tmp/dataset/noframe/train/mask/frame",
        },
        "test": {
            "image": "/root/autodl-tmp/dataset/noframe/test/image",
            "mask": "/root/autodl-tmp/dataset/noframe/test/mask/frame",
        },
    },
    "102": {
        "train": {
            "image": "/root/autodl-tmp/dataset/noframe/train/image",
            "mask": "/root/autodl-tmp/dataset/noframe/train/mask/lens",
        },
        "test": {
            "image": "/root/autodl-tmp/dataset/noframe/test/image",
            "mask": "/root/autodl-tmp/dataset/noframe/test/mask/lens",
        },
    },
}


def modify(dataset_index: str):
    _preprocessed_path = "/root/autodl-tmp/data/_preprocessed"
    datasets = os.listdir(_preprocessed_path)
    dataset = [i for i in datasets if f"Dataset{dataset_index}" in i][0]
    json_file = os.path.join(_preprocessed_path, dataset, "splits_final.json")

    train_path = path_map[dataset_index]["train"]["image"]
    test_path = path_map[dataset_index]["test"]["image"]
    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    train_files = list(set(train_files) - set(test_files))
    train_prefix = [os.path.splitext(i)[0] for i in train_files]
    test_prefix = [os.path.splitext(i)[0] for i in test_files]
    # print(len(train_prefix))
    # print(len(test_prefix))
    data = {"train": train_prefix, "val": test_prefix}
    data = [data] * 5
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    # modify("001")
    modify("002")
    # modify("102")
