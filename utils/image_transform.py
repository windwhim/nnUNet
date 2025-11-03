import os
import cv2
from tqdm import tqdm

from utils import get_files

files = get_files("/root/autodl-tmp/dataset/frame/train/image", ".png")
for file in tqdm(files):

    filename = os.path.basename(file)
    if filename.startswith("2025"):
        img = cv2.imread(file)
        cv2.imwrite(file, img)
