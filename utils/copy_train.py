import os
from utils import copy, get_files


def copy_train(path):
    train_image_path = os.path.join(path, "train", "image")
    train_mask_frame_path = os.path.join(path, "train", "mask", "frame")
    train_mask_lens_path = os.path.join(path, "train", "mask", "lens")

    test_image_path = os.path.join(path, "test", "image")
    test_mask_frame_path = os.path.join(path, "test", "mask", "frame")
    test_mask_lens_path = os.path.join(path, "test", "mask", "lens")
    image_files = get_files(test_image_path, ".png")
    mask_frame_files = get_files(test_mask_frame_path, ".png")
    mask_lens_files = get_files(test_mask_lens_path, ".png")
    # image_files.sort()
    for file in image_files:
        filename = os.path.basename(file)
        copy(
            file,
            os.path.join(train_image_path, filename),
            True,
        )
    for file in mask_frame_files:
        filename = os.path.basename(file)
        copy(
            file,
            os.path.join(train_mask_frame_path, filename),
            True,
        )
    for file in mask_lens_files:
        filename = os.path.basename(file)
        copy(
            file,
            os.path.join(train_mask_lens_path, filename),
            True,
        )


if __name__ == "__main__":
    copy_train("/root/autodl-tmp/dataset/noframe")
    copy_train("/root/autodl-tmp/dataset/frame")
