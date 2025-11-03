import cv2
from utils import get_files


def show_mask(path):
    files = get_files(path, ".png")
    print(len(files))
    for file in files:
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        img[img == 1] = 255
        cv2.imwrite(file, img)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)


if __name__ == "__main__":
    path = "/root/autodl-tmp/predict/NoFrameFrame"
    show_mask(path)
