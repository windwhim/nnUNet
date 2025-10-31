from utils import get_files


def total_nums(path):
    files = get_files(path, ".png")
    print(len(files))


if __name__ == "__main__":
    total_nums("/root/autodl-tmp/dataset/noframe/train/image")
    total_nums("/root/autodl-tmp/dataset/noframe/train/mask/frame")
    total_nums("/root/autodl-tmp/dataset/noframe/train/mask/lens")
    total_nums("/root/autodl-tmp/dataset/noframe/test/image")
    total_nums("/root/autodl-tmp/dataset/noframe/test/mask/frame")
    total_nums("/root/autodl-tmp/dataset/noframe/test/mask/lens")

    total_nums("/root/autodl-tmp/dataset/frame/train/image")
    total_nums("/root/autodl-tmp/dataset/frame/train/mask/frame")
    total_nums("/root/autodl-tmp/dataset/frame/train/mask/lens")

    total_nums("/root/autodl-tmp/dataset/frame/test/image")
    total_nums("/root/autodl-tmp/dataset/frame/test/mask/frame")
    total_nums("/root/autodl-tmp/dataset/frame/test/mask/lens")
