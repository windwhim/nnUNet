import os
from tqdm import tqdm
from utils import get_files, load_types, copy, move


def split_frame():
    base_image_path = "/root/autodl-tmp/dataset/frame/image/"
    base_mask_frame_path = "/root/autodl-tmp/dataset/frame/mask/frame/"
    base_mask_lens_path = "/root/autodl-tmp/dataset/frame/mask/lens/"

    dst_image_path = "/root/autodl-tmp/dataset/noframe/image/"
    dst_mask_frame_path = "/root/autodl-tmp/dataset/noframe/mask/frame/"
    dst_mask_lens_path = "/root/autodl-tmp/dataset/noframe/mask/lens/"

    types = load_types()
    skus = list(types.keys())
    for sku in tqdm(skus):
        frame_type = types[sku]["frame"]
        if frame_type == 0:
            continue
        image_file = os.path.join(base_image_path, sku + "_1.png")
        if not os.path.exists(image_file):
            continue
        mask_frame_file = os.path.join(base_mask_frame_path, sku + "_1.png")
        mask_lens_file = os.path.join(base_mask_lens_path, sku + "_1.png")
        dst_image_file = os.path.join(dst_image_path, sku + "_1.png")
        dst_mask_frame_file = os.path.join(dst_mask_frame_path, sku + "_1.png")
        dst_mask_lens_file = os.path.join(dst_mask_lens_path, sku + "_1.png")
        move(image_file, dst_image_file, True)
        move(mask_frame_file, dst_mask_frame_file, True)
        move(mask_lens_file, dst_mask_lens_file, True)


if __name__ == "__main__":
    split_frame()
