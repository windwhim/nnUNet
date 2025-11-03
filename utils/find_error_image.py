import os
import cv2
from PIL import Image, ImageFile
from tqdm import tqdm
from utils import get_files

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True


def fix_truncated_images(input_path, output_path=None):
    """
    检测并修复截断的图像文件

    Args:
        input_path: 输入图像文件夹路径
        output_path: 输出图像文件夹路径（可选，如果未提供则覆盖原文件）
    """
    if output_path is None:
        output_path = input_path

    # 获取所有PNG文件
    files = get_files(input_path, ".png")

    print(f"处理 {len(files)} 个图像文件...")

    fixed_count = 0
    error_count = 0
    errs = []
    for file in tqdm(files):
        try:
            # 尝试使用PIL加载图像
            img = Image.open(file)
            img.load()  # 这会触发实际加载，如果图像损坏会抛出异常

            # # 如果成功加载，保存图像（可能修复截断问题）
            # output_file = file.replace(input_path, output_path) if output_path != input_path else file
            # os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # # 使用OpenCV重新保存图像
            # img_cv = cv2.imread(file)
            # if img_cv is not None:
            #     cv2.imwrite(output_file, img_cv)
            #     fixed_count += 1
            # else:
            #     print(f"无法使用OpenCV读取: {file}")
            #     error_count += 1

        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            errs.append(file)
            error_count += 1
    print(errs)

    print(f"处理完成: {fixed_count} 个文件已修复, {error_count} 个文件处理失败")


if __name__ == "__main__":
    # 修改为你的数据集路径
    dataset_path = "/root/autodl-tmp/dataset/frame/train/image"
    fix_truncated_images(dataset_path)
