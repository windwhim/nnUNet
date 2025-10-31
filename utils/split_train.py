import os

from tqdm import tqdm
from utils import get_files, move, get_sku, copy
import random


def split_train(path, num):
    train_image_path = os.path.join(path, "train", "image")
    train_mask_frame_path = os.path.join(path, "train", "mask", "frame")
    train_mask_lens_path = os.path.join(path, "train", "mask", "lens")
    # train_image_files = get_files(train_image_path, ".png")
    test_image_path = os.path.join(path, "test", "image")
    test_mask_frame_path = os.path.join(path, "test", "mask", "frame")
    test_mask_lens_path = os.path.join(path, "test", "mask", "lens")

    image_files = get_files(train_image_path, ".png")
    mask_frame_files = get_files(train_mask_frame_path, ".png")
    mask_lens_files = get_files(train_mask_lens_path, ".png")
    # files = image_files + mask_frame_files + mask_lens_files
    image_files.sort()
    mask_frame_files.sort()
    mask_lens_files.sort()
    nums_image, nums_mask_frame, nums_mask_lens = (
        len(image_files),
        len(mask_frame_files),
        len(mask_lens_files),
    )
    assert (
        nums_image == nums_mask_frame == nums_mask_lens
    ), "nums_image != nums_mask_frame != nums_mask_lens"
    selected_index = random.sample(range(nums_image), num)
    for i in tqdm(selected_index):
        image_file = image_files[i]
        mask_frame_file = mask_frame_files[i]
        mask_lens_file = mask_lens_files[i]

        assert (
            get_sku(image_file) == get_sku(mask_frame_file) == get_sku(mask_lens_file)
        ), "sku not equal"
        copy(
            image_file,
            os.path.join(test_image_path, os.path.basename(image_file)),
            True,
        )
        copy(
            mask_frame_file,
            os.path.join(test_mask_frame_path, os.path.basename(mask_frame_file)),
            True,
        )
        copy(
            mask_lens_file,
            os.path.join(test_mask_lens_path, os.path.basename(mask_lens_file)),
            True,
        )
        # move(image_file, os.path.join(test_image_path, os.path.basename(image_file)), True)
        # move(mask_frame_file, os.path.join(test_mask_frame_path, os.path.basename(mask_frame_file)), True)
        # move(mask_lens_file, os.path.join(test_mask_lens_path, os.path.basename(mask_lens_file)), True)

    # print(selected_index)


if __name__ == "__main__":
    # split_train("/root/autodl-tmp/dataset/frame", 567)
    split_train("/root/autodl-tmp/dataset/noframe", 147)
