import os

import cv2


import shutil


def get_files(path, ext=None):
    if isinstance(ext, str):
        ext = (ext,)
    files = []

    for root, dirs, fs in os.walk(path):
        for filename in fs:
            files.append(os.path.join(root, filename))
    if ext is not None:
        files = [f for f in files if os.path.splitext(f)[1] in ext]
    return files


def copy(src, dst, force=False):
    if not force and os.path.exists(dst):
        return
    path = os.path.dirname(dst)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    shutil.copy(src, dst)


def move(src, dst, force=False):
    if not force and os.path.exists(dst):
        return
    path = os.path.dirname(dst)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    shutil.move(src, dst)


def get_sku(file_path):
    # 返回sku
    sku = os.path.basename(file_path).split("_")[0]
    return sku


def get_view(file_path) -> int:
    # 返回视图
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    view = filename_without_ext.split("_")[1]
    return int(view)


def is_foreground(file_path):
    # 是否为背景
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    return False if filename_without_ext.endswith("bg") else True


def save_image(path, image):
    """
    保存图像到指定路径，如果路径不存在则创建。

    参数:
    image: 图像数据 (numpy.ndarray)
    path: 图像保存的完整路径包括文件名，如 './images/new_image.jpg'
    """
    # 分离路径和文件名
    directory, filename = os.path.split(path)

    # 检查目录是否存在，如果不存在，则创建
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    # 使用OpenCV保存图像
    cv2.imwrite(path, image)


def load_types(path=None):
    if path is None:
        path = "/root/code/nnUNet/utils/types4.txt"
    with open(path, "r") as f:
        data = f.readlines()
        data = [d.strip().split("    ") for d in data]
    types = {}
    for d in data:
        sku = d[0]
        params = d[1].split(",")
        params = [int(p) for p in params]

        if len(params) == 4:
            params_dict = {
                "frame": int(params[0]),
                "pilot": int(params[1]),
                "material": int(params[2]),
                "transparent": int(params[3]),
            }
        elif len(params) == 3:
            params_dict = {
                "frame": int(params[0]),
                "pilot": int(params[1]),
                "material": int(params[2]),
            }
        elif len(params) == 2:
            params_dict = {"frame": int(params[0]), "pilot": int(params[1])}
        elif len(params) == 1:
            params_dict = {"frame": int(params[0])}
        else:
            raise Exception("Invalid params length")
        types[sku] = params_dict
    return types
