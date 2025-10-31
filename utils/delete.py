import os
from utils import get_files, get_sku


def delete():
    noframe_image_path = "/root/autodl-tmp/dataset/noframe/train/image/"
    noframe_mask_frame_path = "/root/autodl-tmp/dataset/noframe/train/mask/frame/"
    noframe_mask_lens_path = "/root/autodl-tmp/dataset/noframe/train/mask/lens/"
    image_files = get_files(noframe_image_path, ".png")
    image_skus = [get_sku(file) for file in image_files]
    mask_frame_files = get_files(noframe_mask_frame_path, ".png")
    mask_frame_skus = [get_sku(file) for file in mask_frame_files]
    mask_lens_files = get_files(noframe_mask_lens_path, ".png")
    mask_lens_skus = [get_sku(file) for file in mask_lens_files]
    # assert len(image_skus) == len(mask_frame_skus) == len(mask_lens_skus), "sku not equal"
    # mask_lens_files = get_files(noframe_mask_lens_path, ".png")
    # image_files.sort()
    # mask_frame_files.sort()
    # mask_lens_files.sort()

    frame_image_path = "/root/autodl-tmp/dataset/frame/train/image/"
    frame_mask_frame_path = "/root/autodl-tmp/dataset/frame/train/mask/frame/"
    frame_mask_lens_path = "/root/autodl-tmp/dataset/frame/train/mask/lens/"
    frame_image_files = get_files(frame_image_path, ".png")
    frame_mask_frame_files = get_files(frame_mask_frame_path, ".png")
    frame_mask_lens_files = get_files(frame_mask_lens_path, ".png")

    for file in frame_image_files:
        sku = get_sku(file)
        if sku in image_skus:
            os.remove(file)

    for file in frame_mask_frame_files:
        sku = get_sku(file)
        if sku in mask_frame_skus:
            os.remove(file)

    for file in frame_mask_lens_files:
        sku = get_sku(file)
        if sku in mask_lens_skus:
            os.remove(file)


def delete2():
    noframe_image_path = "/root/autodl-tmp/dataset/noframe/test/image/"
    noframe_mask_frame_path = "/root/autodl-tmp/dataset/noframe/test/mask/frame/"
    noframe_mask_lens_path = "/root/autodl-tmp/dataset/noframe/test/mask/lens/"
    image_files = get_files(noframe_image_path, ".png")
    image_skus = [get_sku(file) for file in image_files]
    mask_frame_files = get_files(noframe_mask_frame_path, ".png")
    mask_frame_skus = [get_sku(file) for file in mask_frame_files]
    mask_lens_files = get_files(noframe_mask_lens_path, ".png")
    mask_lens_skus = [get_sku(file) for file in mask_lens_files]
    # assert len(image_skus) == len(mask_frame_skus) == len(mask_lens_skus), "sku not equal"
    # mask_lens_files = get_files(noframe_mask_lens_path, ".png")
    # image_files.sort()
    # mask_frame_files.sort()
    # mask_lens_files.sort()

    frame_image_path = "/root/autodl-tmp/dataset/noframe/train/image/"
    frame_mask_frame_path = "/root/autodl-tmp/dataset/noframe/train/mask/frame/"
    frame_mask_lens_path = "/root/autodl-tmp/dataset/noframe/train/mask/lens/"
    frame_image_files = get_files(frame_image_path, ".png")
    frame_mask_frame_files = get_files(frame_mask_frame_path, ".png")
    frame_mask_lens_files = get_files(frame_mask_lens_path, ".png")

    for file in frame_image_files:
        sku = get_sku(file)
        if sku in image_skus:
            os.remove(file)

    for file in frame_mask_frame_files:
        sku = get_sku(file)
        if sku in mask_frame_skus:
            os.remove(file)

    for file in frame_mask_lens_files:
        sku = get_sku(file)
        if sku in mask_lens_skus:
            os.remove(file)


delete2()
