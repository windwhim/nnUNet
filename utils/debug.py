from utils import get_files, copy
import os


def debug():
    base_image_path = "/root/autodl-tmp/dataset/noframe/train/image/"
    base_mask_frame_path = "/root/autodl-tmp/dataset/noframe/train/mask/frame/"
    base_mask_lens_path = "/root/autodl-tmp/dataset/noframe/train/mask/lens/"
    dst_image_path = "/root/autodl-tmp/dataset/frame/train/image/"
    dst_mask_frame_path = "/root/autodl-tmp/dataset/frame/train/mask/frame/"
    dst_mask_lens_path = "/root/autodl-tmp/dataset/frame/train/mask/lens/"
    image_files = get_files(base_image_path, ".png")
    mask_frame_files = get_files(base_mask_frame_path, ".png")
    mask_lens_files = get_files(base_mask_lens_path, ".png")
    image_files.sort()
    mask_frame_files.sort()
    mask_lens_files.sort()

    for file in image_files:
        filename = os.path.basename(file)
        dst_file = os.path.join(dst_image_path, filename)

        copy(file, dst_file)

    for file in mask_frame_files:
        filename = os.path.basename(file)
        dst_file = os.path.join(dst_mask_frame_path, filename)

        copy(file, dst_file)

    for file in mask_lens_files:
        filename = os.path.basename(file)
        dst_file = os.path.join(dst_mask_lens_path, filename)

        copy(file, dst_file)


debug()
